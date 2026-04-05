import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

from conformer.conformer.encoder import ConformerBlock
from conformer.conformer.modules import Linear
from conformer.conformer.decoder import LSTMAttentionDecoder, BeamSearchLSTM


class Connector(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2
                         , 1)

  def forward(self, x, x_lengths):
    x = x * math.sqrt(self.hidden_channels) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask



class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class SpeechEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs




class SpeechDecoder(nn.Module):
  """
  From EEG to Wav for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size


    self.enc_proj = Connector(
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes)
    self.enc_q = SpeechEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4)

    # Add subject discriminator for speech latent
    from models import SubjectDiscriminator  # absolute import for script usage
    self.subject_discriminator = SubjectDiscriminator(inter_channels, hidden_channels, kwargs.get('num_subjects', 25))
    # Add subject discriminator for speech latent (separate from EEG)
    self.speech_subject_discriminator = SubjectDiscriminator(inter_channels, hidden_channels, kwargs.get('num_subjects', 25))
    # Add phoneme predictor for speech latent
    self.speech_phoneme_predictor = PhonemePredictor(
        num_layers=kwargs.get('num_conformer_layers', 0),
        num_class=kwargs.get('vocab_size', 51),
        encoder_dim=hidden_channels
    )

  def forward(self, x, x_lengths, y, y_lengths, alpha=1.0):
    x, m_p, logs_p, x_mask = self.enc_proj(x, x_lengths)
    g=None
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)
    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    # Subject discriminator on latent (z or z_p)
    z_for_subj = z
    # print('SpeechDecoder: z shape before subject discriminator:', z_for_subj.shape)
    if z_for_subj.dim() == 3:
        # [batch, inter_channels, time] -> mean over time
        z_for_subj = z_for_subj.mean(dim=2)
    if z_for_subj.shape[1] == 1:
        print('WARNING: z_for_subj has only 1 channel, squeezing. This may indicate a config or model bug.')
        z_for_subj = z_for_subj.squeeze(1)
    # print('SpeechDecoder: z_for_subj shape for subject discriminator:', z_for_subj.shape)
    assert z_for_subj.dim() == 2, f"SubjectDiscriminator input must be 2D, got {z_for_subj.shape}"
    sid_logits_speech = self.speech_subject_discriminator(z_for_subj, alpha)
    z_for_phoneme = z
    if z_for_phoneme.dim() == 3:
        z_for_phoneme_lengths = y_lengths
        # Predict phonemes from speech latent
        speech_phoneme_preds = self.speech_phoneme_predictor(z_for_phoneme, z_for_phoneme_lengths, None)
    else:
        speech_phoneme_preds = self.speech_phoneme_predictor(z_for_phoneme.unsqueeze(2), y_lengths, None)
    return o, None, None, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), sid_logits_speech, speech_phoneme_preds

  def infer(self, x, x_lengths, noise_scale=0.667, max_len=None, alpha=1.0):
    x, m_p, logs_p, x_mask = self.enc_proj(x, x_lengths)
    g = None

    y_lengths = x_lengths.clone()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

    m_p=m_p[:,:,:y_mask.size(2)]
    logs_p=logs_p[:,:,:y_mask.size(2)]

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    z_for_subj = z
    if z_for_subj.dim() == 3:
        z_for_subj = z_for_subj.mean(dim=2)
    sid_logits_speech = self.speech_subject_discriminator(z_for_subj, alpha)
    z_for_phoneme = z
    if z_for_phoneme.dim() == 3:
        z_for_phoneme_lengths = x_lengths
        speech_phoneme_preds = self.speech_phoneme_predictor(z_for_phoneme, z_for_phoneme_lengths, None)
    else:
        speech_phoneme_preds = self.speech_phoneme_predictor(z_for_phoneme.unsqueeze(2), x_lengths, None)
    return o, None, y_mask, (z, z_p, m_p, logs_p), sid_logits_speech, speech_phoneme_preds
  

class PhonemePredictor(nn.Module):
  def __init__(
        self, num_layers=1,
        num_class=51,
        max_length=50,
        encoder_dim=192,
        activate_decoder=True

       ):
    super(PhonemePredictor, self).__init__()

    self.conformer_blocks = nn.ModuleList([ConformerBlock() for _ in range(num_layers)])
    self.fc = Linear(encoder_dim, num_class, bias=False) # conformer dim, vocab size
    self.decoder = LSTMAttentionDecoder( # adopted from https://github.com/openspeech-team/openspeech/blob/main/openspeech/models/conformer/model.py
            num_classes=num_class,
            max_length=max_length,
            hidden_state_dim=encoder_dim,
            pad_id=0,
            sos_id=1,
            eos_id=2,
            num_heads=4,
            dropout_p=0.3,
            num_layers=1,
            attn_mechanism='loc',
            rnn_type='lstm',
        )
    self.activate_decoder = activate_decoder


  def set_beam_decoder(self, beam_size: int = 3):
    """Setting beam search decoder"""
    self.beam_decoder = BeamSearchLSTM(
        decoder=self.decoder.cuda(),
        beam_size=beam_size,
    ).cuda()

  def forward(self, x, x_lengths, phn_labels):

    x = x.transpose(2,1)


    for i,block in enumerate(self.conformer_blocks):
      x = block(x)
    
    if self.activate_decoder:
        x = self.decoder(encoder_outputs=x, targets=phn_labels, encoder_output_lengths=x_lengths, teacher_forcing_ratio=0.0)
    
    return x
  
  def topk_search(self, x, x_lengths, phn_labels, topk=5):

    x = x.transpose(2,1)
    for i, block in enumerate(self.conformer_blocks):
       x = block(x)
    
    topk_preds = self.beam_decoder(x, x_lengths)

    return topk_preds
  
  def get_next_topk(self, x, x_lengths, phn_labels, topk=5):

    x = x.transpose(2,1)
    for i, block in enumerate(self.conformer_blocks):
       x = block(x)
    
    topk_preds = self.decoder.next_topk(x, x_lengths, phn_labels)
    topk_preds = torch.topk(topk_preds, topk, dim=2)[1]

    return topk_preds
         

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SubjectDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_subjects, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_channels // 2, num_subjects)
        )

    def forward(self, x, alpha=1.0):
        x = GradientReversalFunction.apply(x, alpha)
        return self.net(x)


