//! Source-authenticated Zonos prefix conditioners.
//!
//! This module contains only the seven typed conditioner transforms from the
//! official `zonos/conditioning.py`. It does not invoke eSpeak; callers supply
//! its audited symbol ids in [`super::ZonosConditioningPacket`].

use super::ZonosConditioningPacket;
use crate::compute::Compute;
use vokra_core::{Result, VokraError};

/// Strictly bound seven-conditioner weight set.
#[derive(Debug, Clone)]
pub struct ZonosPrefixConditionerWeights {
    /// eSpeak symbol embedding, `[189, d_model]`.
    phoneme_embedder: Vec<f32>,
    /// Speaker projection in GEMM layout `[128, d_model]`.
    speaker_project: Vec<f32>,
    /// Speaker learned unconditional vector, `[d_model]`.
    speaker_uncond: Vec<f32>,
    /// Emotion Fourier frequencies in GEMM layout `[8, 1024]`.
    emotion_weight: Vec<f32>,
    /// Emotion learned unconditional vector, `[d_model]`.
    emotion_uncond: Vec<f32>,
    /// fmax Fourier frequencies in GEMM layout `[1, 1024]`.
    fmax_weight: Vec<f32>,
    /// fmax learned unconditional vector, `[d_model]`.
    fmax_uncond: Vec<f32>,
    /// Pitch standard deviation frequencies in GEMM layout `[1, 1024]`.
    pitch_std_weight: Vec<f32>,
    /// Pitch standard deviation learned unconditional vector, `[d_model]`.
    pitch_std_uncond: Vec<f32>,
    /// Speaking-rate frequencies in GEMM layout `[1, 1024]`.
    speaking_rate_weight: Vec<f32>,
    /// Speaking-rate learned unconditional vector, `[d_model]`.
    speaking_rate_uncond: Vec<f32>,
    /// Language integer embedding, `[128, d_model]`.
    language_embedder: Vec<f32>,
    /// Language learned unconditional vector, `[d_model]`.
    language_uncond: Vec<f32>,
    /// Speaker projection bias, `[d_model]`.
    speaker_bias: Vec<f32>,
    /// Prefix-wide projection in GEMM layout `[d_model, d_model]`.
    project: Vec<f32>,
    /// Prefix-wide projection bias, `[d_model]`.
    project_bias: Vec<f32>,
    /// Prefix-wide LayerNorm gamma, `[d_model]`.
    norm_weight: Vec<f32>,
    /// Prefix-wide LayerNorm beta, `[d_model]`.
    norm_bias: Vec<f32>,
}

/// Constructor-only packet used by the strict GGUF binder. Keeping the
/// fields crate-private prevents callers from fabricating a native weight
/// set through public struct literals.
pub(crate) struct ZonosPrefixConditionerParts {
    pub(crate) phoneme_embedder: Vec<f32>,
    pub(crate) speaker_project: Vec<f32>,
    pub(crate) speaker_uncond: Vec<f32>,
    pub(crate) emotion_weight: Vec<f32>,
    pub(crate) emotion_uncond: Vec<f32>,
    pub(crate) fmax_weight: Vec<f32>,
    pub(crate) fmax_uncond: Vec<f32>,
    pub(crate) pitch_std_weight: Vec<f32>,
    pub(crate) pitch_std_uncond: Vec<f32>,
    pub(crate) speaking_rate_weight: Vec<f32>,
    pub(crate) speaking_rate_uncond: Vec<f32>,
    pub(crate) language_embedder: Vec<f32>,
    pub(crate) language_uncond: Vec<f32>,
    pub(crate) speaker_bias: Vec<f32>,
    pub(crate) project: Vec<f32>,
    pub(crate) project_bias: Vec<f32>,
    pub(crate) norm_weight: Vec<f32>,
    pub(crate) norm_bias: Vec<f32>,
}

impl ZonosPrefixConditionerWeights {
    pub(crate) fn validate(&self, d_model: usize) -> Result<()> {
        if d_model != 2048
            || self.phoneme_embedder.len() != 189 * d_model
            || self.speaker_project.len() != 128 * d_model
            || self.speaker_uncond.len() != d_model
            || self.emotion_weight.len() != 8 * 1024
            || self.emotion_uncond.len() != d_model
            || self.fmax_weight.len() != 1024
            || self.fmax_uncond.len() != d_model
            || self.pitch_std_weight.len() != 1024
            || self.pitch_std_uncond.len() != d_model
            || self.speaking_rate_weight.len() != 1024
            || self.speaking_rate_uncond.len() != d_model
            || self.language_embedder.len() != 128 * d_model
            || self.language_uncond.len() != d_model
            || self.speaker_bias.len() != d_model
            || self.project.len() != d_model * d_model
            || self.project_bias.len() != d_model
            || self.norm_weight.len() != d_model
            || self.norm_bias.len() != d_model
        {
            return Err(VokraError::InvalidArgument(
                "zonos prefix conditioner weight shape mismatch".to_owned(),
            ));
        }
        let all = [
            &self.phoneme_embedder,
            &self.speaker_project,
            &self.speaker_uncond,
            &self.emotion_weight,
            &self.emotion_uncond,
            &self.fmax_weight,
            &self.fmax_uncond,
            &self.pitch_std_weight,
            &self.pitch_std_uncond,
            &self.speaking_rate_weight,
            &self.speaking_rate_uncond,
            &self.language_embedder,
            &self.language_uncond,
            &self.speaker_bias,
            &self.project,
            &self.project_bias,
            &self.norm_weight,
            &self.norm_bias,
        ];
        if all
            .iter()
            .any(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(VokraError::InvalidArgument(
                "zonos prefix weights contain non-finite values".to_owned(),
            ));
        }
        Ok(())
    }

    pub(crate) fn from_parts(parts: ZonosPrefixConditionerParts) -> Result<Self> {
        let fields = [
            ("phoneme_embedder", parts.phoneme_embedder.len(), 189 * 2048),
            ("speaker_project", parts.speaker_project.len(), 128 * 2048),
            ("speaker_uncond", parts.speaker_uncond.len(), 2048),
            ("emotion_weight", parts.emotion_weight.len(), 8 * 1024),
            ("emotion_uncond", parts.emotion_uncond.len(), 2048),
            ("fmax_weight", parts.fmax_weight.len(), 1024),
            ("fmax_uncond", parts.fmax_uncond.len(), 2048),
            ("pitch_std_weight", parts.pitch_std_weight.len(), 1024),
            ("pitch_std_uncond", parts.pitch_std_uncond.len(), 2048),
            (
                "speaking_rate_weight",
                parts.speaking_rate_weight.len(),
                1024,
            ),
            (
                "speaking_rate_uncond",
                parts.speaking_rate_uncond.len(),
                2048,
            ),
            (
                "language_embedder",
                parts.language_embedder.len(),
                128 * 2048,
            ),
            ("language_uncond", parts.language_uncond.len(), 2048),
            ("speaker_bias", parts.speaker_bias.len(), 2048),
            ("project", parts.project.len(), 2048 * 2048),
            ("project_bias", parts.project_bias.len(), 2048),
            ("norm_weight", parts.norm_weight.len(), 2048),
            ("norm_bias", parts.norm_bias.len(), 2048),
        ];
        if let Some((name, actual, expected)) = fields.into_iter().find(|(_, a, e)| a != e) {
            return Err(VokraError::InvalidArgument(format!(
                "zonos prefix weight `{name}` has {actual} values, expected {expected}"
            )));
        }
        let all = [
            &parts.phoneme_embedder,
            &parts.speaker_project,
            &parts.speaker_uncond,
            &parts.emotion_weight,
            &parts.emotion_uncond,
            &parts.fmax_weight,
            &parts.fmax_uncond,
            &parts.pitch_std_weight,
            &parts.pitch_std_uncond,
            &parts.speaking_rate_weight,
            &parts.speaking_rate_uncond,
            &parts.language_embedder,
            &parts.language_uncond,
            &parts.speaker_bias,
            &parts.project,
            &parts.project_bias,
            &parts.norm_weight,
            &parts.norm_bias,
        ];
        if all
            .iter()
            .any(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(VokraError::InvalidArgument(
                "zonos prefix weights contain non-finite values".to_owned(),
            ));
        }
        let result = Self {
            phoneme_embedder: parts.phoneme_embedder,
            speaker_project: parts.speaker_project,
            speaker_uncond: parts.speaker_uncond,
            emotion_weight: parts.emotion_weight,
            emotion_uncond: parts.emotion_uncond,
            fmax_weight: parts.fmax_weight,
            fmax_uncond: parts.fmax_uncond,
            pitch_std_weight: parts.pitch_std_weight,
            pitch_std_uncond: parts.pitch_std_uncond,
            speaking_rate_weight: parts.speaking_rate_weight,
            speaking_rate_uncond: parts.speaking_rate_uncond,
            language_embedder: parts.language_embedder,
            language_uncond: parts.language_uncond,
            speaker_bias: parts.speaker_bias,
            project: parts.project,
            project_bias: parts.project_bias,
            norm_weight: parts.norm_weight,
            norm_bias: parts.norm_bias,
        };
        result.validate(2048)?;
        Ok(result)
    }
}

/// Builds conditional and CFG-unconditional prefix sequences from raw typed
/// controls. The unconditional branch keeps eSpeak (a required key) and uses
/// each learned unconditional vector for the other six conditioners.
pub(crate) fn build_prefix(
    packet: &ZonosConditioningPacket,
    weights: &ZonosPrefixConditionerWeights,
    compute: &Compute,
    d_model: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    weights.validate(d_model)?;
    if d_model != 2048 || packet.speaker.len() != 128 || packet.emotion.len() != 8 {
        return Err(VokraError::InvalidArgument(
            "zonos conditioning controls have invalid native shapes".to_owned(),
        ));
    }
    validate_phoneme_ids(&packet.phoneme_ids)?;
    if packet.speaker.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos speaker controls contain non-finite values".to_owned(),
        ));
    }
    let normalized_emotion = normalize_emotion(&packet.emotion)?;
    if !packet.fmax.is_finite()
        || !packet.pitch_std.is_finite()
        || !packet.speaking_rate.is_finite()
        || !(0.0..=24_000.0).contains(&packet.fmax)
        || !(0.0..=400.0).contains(&packet.pitch_std)
        || !(0.0..=40.0).contains(&packet.speaking_rate)
        || !(-1..=126).contains(&packet.language_id)
    {
        return Err(VokraError::InvalidArgument(
            "zonos conditioning control is outside the audited domain".to_owned(),
        ));
    }
    let prefix_values = packet
        .phoneme_ids
        .len()
        .checked_add(6)
        .and_then(|rows| rows.checked_mul(d_model))
        .filter(|&values| values <= (1 << 24))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "zonos conditioning prefix is larger than the bounded native buffer".to_owned(),
            )
        })?;
    let mut conditional = Vec::with_capacity(prefix_values);
    let mut unconditional = Vec::with_capacity(prefix_values);
    for &symbol in &packet.phoneme_ids {
        let symbol = symbol as usize;
        if symbol >= 189 {
            return Err(VokraError::InvalidArgument(
                "zonos phoneme symbol is outside the authenticated 189-entry table".to_owned(),
            ));
        }
        let row = &weights.phoneme_embedder[symbol * d_model..(symbol + 1) * d_model];
        conditional.extend_from_slice(row);
        // `espeak` is required by the official PrefixConditioner and has no
        // learned unconditional token.
        unconditional.extend_from_slice(row);
    }
    let mut speaker = vec![0.0; d_model];
    compute.gemm_f32(
        1,
        d_model,
        128,
        &packet.speaker,
        &weights.speaker_project,
        Some(&weights.speaker_bias),
        &mut speaker,
    )?;
    conditional.extend_from_slice(&speaker);
    unconditional.extend_from_slice(&weights.speaker_uncond);
    append_fourier(
        &normalized_emotion,
        &weights.emotion_weight,
        &mut conditional,
        d_model,
        compute,
    )?;
    unconditional.extend_from_slice(&weights.emotion_uncond);
    append_fourier_scalar(
        packet.fmax,
        24_000.0,
        &weights.fmax_weight,
        &mut conditional,
        d_model,
        compute,
    )?;
    unconditional.extend_from_slice(&weights.fmax_uncond);
    append_fourier_scalar(
        packet.pitch_std,
        400.0,
        &weights.pitch_std_weight,
        &mut conditional,
        d_model,
        compute,
    )?;
    unconditional.extend_from_slice(&weights.pitch_std_uncond);
    append_fourier_scalar(
        packet.speaking_rate,
        40.0,
        &weights.speaking_rate_weight,
        &mut conditional,
        d_model,
        compute,
    )?;
    unconditional.extend_from_slice(&weights.speaking_rate_uncond);
    let language = usize::try_from(packet.language_id.checked_add(1).ok_or_else(|| {
        VokraError::InvalidArgument("zonos language id cannot be represented".to_owned())
    })?)
    .map_err(|_| {
        VokraError::InvalidArgument("zonos language id cannot be represented".to_owned())
    })?;
    if language >= 128 {
        return Err(VokraError::InvalidArgument(
            "zonos language id is outside the authenticated 128-entry table".to_owned(),
        ));
    }
    conditional.extend_from_slice(
        &weights.language_embedder[language * d_model..(language + 1) * d_model],
    );
    unconditional.extend_from_slice(&weights.language_uncond);
    let conditional = project_prefix(&conditional, weights, d_model, compute)?;
    let unconditional = project_prefix(&unconditional, weights, d_model, compute)?;
    if conditional
        .iter()
        .chain(&unconditional)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "zonos prefix conditioner output is non-finite".to_owned(),
        ));
    }
    Ok((conditional, unconditional))
}

fn normalize_emotion(values: &[f32]) -> Result<Vec<f32>> {
    if values.len() != 8
        || values
            .iter()
            .any(|&value| !value.is_finite() || value < 0.0)
    {
        return Err(VokraError::InvalidArgument(
            "zonos emotion controls must be finite and non-negative".to_owned(),
        ));
    }
    let sum = values.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "zonos emotion controls must have a positive finite sum".to_owned(),
        ));
    }
    let normalized: Vec<f32> = values.iter().map(|value| value / sum).collect();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos normalized emotion controls are non-finite".to_owned(),
        ));
    }
    Ok(normalized)
}

fn validate_phoneme_ids(ids: &[u32]) -> Result<()> {
    if ids.len() < 2
        || ids.first() != Some(&2)
        || ids.last() != Some(&3)
        || ids[1..ids.len() - 1]
            .iter()
            .any(|&id| id == 0 || (2..=3).contains(&id) || id >= 189)
    {
        return Err(VokraError::InvalidArgument(
            "zonos phoneme symbols must use PAD=0/UNK=1/BOS=2/EOS=3 framing".to_owned(),
        ));
    }
    Ok(())
}

fn append_fourier(
    values: &[f32],
    weight: &[f32],
    output: &mut Vec<f32>,
    d_model: usize,
    compute: &Compute,
) -> Result<()> {
    if weight.len() != values.len() * 1024 {
        return Err(VokraError::InvalidArgument(
            "zonos emotion Fourier weight shape mismatch".to_owned(),
        ));
    }
    let mut frequencies = vec![0.0; 1024];
    compute.gemm_f32(
        1,
        1024,
        values.len(),
        values,
        weight,
        None,
        &mut frequencies,
    )?;
    for value in &mut frequencies {
        *value *= 2.0 * std::f32::consts::PI;
    }
    if frequencies.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos emotion Fourier projection is non-finite".to_owned(),
        ));
    }
    output.extend(frequencies.iter().map(|value| value.cos()));
    output.extend(frequencies.iter().map(|value| value.sin()));
    debug_assert_eq!(output.len() % d_model, 0);
    Ok(())
}

fn append_fourier_scalar(
    value: f32,
    max_value: f32,
    weight: &[f32],
    output: &mut Vec<f32>,
    d_model: usize,
    compute: &Compute,
) -> Result<()> {
    if weight.len() != 1024 {
        return Err(VokraError::InvalidArgument(
            "zonos scalar Fourier weight shape mismatch".to_owned(),
        ));
    }
    let mut frequencies = vec![0.0; 1024];
    compute.gemm_f32(
        1,
        1024,
        1,
        &[value / max_value],
        weight,
        None,
        &mut frequencies,
    )?;
    for frequency in &mut frequencies {
        *frequency *= 2.0 * std::f32::consts::PI;
    }
    if frequencies.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "zonos scalar Fourier projection is non-finite".to_owned(),
        ));
    }
    output.extend(frequencies.iter().map(|value| value.cos()));
    output.extend(frequencies.iter().map(|value| value.sin()));
    debug_assert_eq!(output.len() % d_model, 0);
    Ok(())
}

fn project_prefix(
    input: &[f32],
    weights: &ZonosPrefixConditionerWeights,
    d_model: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() % d_model != 0 {
        return Err(VokraError::InvalidArgument(
            "zonos prefix conditioner sequence shape mismatch".to_owned(),
        ));
    }
    let rows = input.len() / d_model;
    let mut projected = vec![0.0; input.len()];
    compute.gemm_f32(
        rows,
        d_model,
        d_model,
        input,
        &weights.project,
        Some(&weights.project_bias),
        &mut projected,
    )?;
    let mut normalized = vec![0.0; projected.len()];
    compute.layer_norm_f32(
        &projected,
        &mut normalized,
        rows,
        d_model,
        &weights.norm_weight,
        &weights.norm_bias,
        1.0e-5,
    )?;
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_shape_and_unconditional_order_are_bounded() {
        let weights = ZonosPrefixConditionerWeights {
            phoneme_embedder: vec![0.0; 189 * 2048],
            speaker_project: vec![0.0; 128 * 2048],
            speaker_uncond: vec![0.0; 2048],
            emotion_weight: vec![0.0; 8 * 1024],
            emotion_uncond: vec![0.0; 2048],
            fmax_weight: vec![0.0; 1024],
            fmax_uncond: vec![0.0; 2048],
            pitch_std_weight: vec![0.0; 1024],
            pitch_std_uncond: vec![0.0; 2048],
            speaking_rate_weight: vec![0.0; 1024],
            speaking_rate_uncond: vec![0.0; 2048],
            language_embedder: vec![0.0; 128 * 2048],
            language_uncond: vec![0.0; 2048],
            speaker_bias: vec![0.0; 2048],
            project: vec![0.0; 2048 * 2048],
            project_bias: vec![0.0; 2048],
            norm_weight: vec![1.0; 2048],
            norm_bias: vec![0.0; 2048],
        };
        let packet = ZonosConditioningPacket {
            version: 1,
            phoneme_ids: vec![2, 42, 3],
            speaker: vec![0.0; 128],
            emotion: vec![0.125; 8],
            fmax: 22_050.0,
            pitch_std: 20.0,
            speaking_rate: 15.0,
            language_id: 0,
            prompt_codes: vec![],
            digest: [0; 32],
        };
        let compute = Compute::cpu();
        let (conditional, unconditional) = build_prefix(&packet, &weights, &compute, 2048).unwrap();
        assert_eq!(conditional.len(), 9 * 2048);
        assert_eq!(unconditional.len(), 9 * 2048);
        assert!(conditional.iter().all(|value| value.is_finite()));
        assert!(unconditional.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn emotion_is_normalized_before_fourier_and_scalar_oracle_is_stable() {
        let weights = vec![1.0; 8 * 1024];
        let compute = Compute::cpu();
        let normalized = normalize_emotion(&[2.0; 8]).unwrap();
        assert!(
            normalized
                .iter()
                .all(|value| (*value - 0.125).abs() < 1.0e-6)
        );
        let mut output = Vec::new();
        append_fourier(&normalized, &weights, &mut output, 2048, &compute).unwrap();
        // Each frequency is 1, so the source Fourier phase is 2π: cos=1,
        // sin=0. This is an explicit deterministic oracle for the transform.
        assert!((output[0] - 1.0).abs() < 1.0e-5);
        assert!(output[1024].abs() < 1.0e-5);

        let scalar_weights = vec![1.0; 1024];
        let mut scalar = Vec::new();
        append_fourier_scalar(20.0, 40.0, &scalar_weights, &mut scalar, 2048, &compute).unwrap();
        assert!((scalar[0] + 1.0).abs() < 1.0e-5);
        assert!(scalar[1024].abs() < 1.0e-5);
    }

    #[test]
    fn emotion_and_phoneme_contracts_fail_closed() {
        assert!(normalize_emotion(&[0.0; 8]).is_err());
        assert!(normalize_emotion(&[f32::NAN; 8]).is_err());
        assert!(validate_phoneme_ids(&[2, 1, 3]).is_ok());
        assert!(validate_phoneme_ids(&[2, 0, 3]).is_err());
        assert!(validate_phoneme_ids(&[2, 2, 3]).is_err());
        assert!(validate_phoneme_ids(&[2, 3, 3]).is_err());

        let packet = ZonosConditioningPacket {
            version: 1,
            phoneme_ids: vec![2, 1, 3],
            speaker: vec![0.0; 128],
            emotion: vec![0.125; 8],
            fmax: 22_050.0,
            pitch_std: 20.0,
            speaking_rate: 15.0,
            language_id: 0,
            prompt_codes: vec![],
            digest: [0; 32],
        };
        let parts = ZonosPrefixConditionerParts {
            phoneme_embedder: vec![0.0; 189 * 2048],
            speaker_project: vec![0.0; 128 * 2048],
            speaker_uncond: vec![0.0; 2048],
            emotion_weight: vec![0.0; 8 * 1024],
            emotion_uncond: vec![0.0; 2048],
            fmax_weight: vec![0.0; 1024],
            fmax_uncond: vec![0.0; 2048],
            pitch_std_weight: vec![0.0; 1024],
            pitch_std_uncond: vec![0.0; 2048],
            speaking_rate_weight: vec![0.0; 1024],
            speaking_rate_uncond: vec![0.0; 2048],
            language_embedder: vec![0.0; 128 * 2048],
            language_uncond: vec![0.0; 2048],
            speaker_bias: vec![0.0; 2048],
            project: vec![0.0; 2048 * 2048],
            project_bias: vec![0.0; 2048],
            norm_weight: vec![1.0; 2048],
            norm_bias: vec![0.0; 2048],
        };
        let weights = ZonosPrefixConditionerWeights::from_parts(parts).unwrap();
        assert!(build_prefix(&packet, &weights, &Compute::cpu(), 2048).is_ok());
    }
}
