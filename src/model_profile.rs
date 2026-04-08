/// Model metadata: pricing, context limits, capabilities, and effective ceilings.
///
/// Used by the cost estimator, compaction thresholds, and capability gating.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    pub name: String,
    /// Input price per 1M tokens (USD)
    pub input_price: f64,
    /// Cached input price per 1M tokens (USD)
    pub cached_input_price: f64,
    /// Output price per 1M tokens (USD)
    pub output_price: f64,
    /// Raw context window in tokens
    pub context_window: usize,
    /// Conservative "smart zone" cap — compaction threshold is 60% of this
    pub effective_ceiling: usize,
    /// Whether the model accepts `reasoning: { effort }` in requests
    pub supports_reasoning_effort_control: bool,
    /// Whether the model returns plaintext reasoning content (e.g. grok-3-mini)
    pub returns_plaintext_reasoning: bool,
    /// Whether the model returns encrypted/opaque reasoning (e.g. grok-4 reasoning)
    pub returns_encrypted_reasoning: bool,
    /// Whether the model supports tool/function calling
    pub supports_tools: bool,
}

impl ModelProfile {
    /// Look up a model by name. Returns known profiles for grok models,
    /// sensible defaults for anything else.
    ///
    /// Resolution order matters — more specific patterns are checked first
    /// (e.g. "grok-3-mini" before "grok-3"). Pattern matching uses
    /// `starts_with` / `contains` so date suffixes like `-0309` resolve
    /// automatically.
    pub fn for_model(name: &str) -> ModelProfile {
        // --- grok-3 family ---
        if name.starts_with("grok-3-mini") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 0.30,
                cached_input_price: 0.05,
                output_price: 0.50,
                context_window: 131_072,
                effective_ceiling: 80_000,
                supports_reasoning_effort_control: true,
                returns_plaintext_reasoning: true,
                returns_encrypted_reasoning: false,
                supports_tools: true,
            };
        }
        if name.starts_with("grok-3") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 3.00,
                cached_input_price: 0.50,
                output_price: 15.00,
                context_window: 131_072,
                effective_ceiling: 80_000,
                supports_reasoning_effort_control: false,
                returns_plaintext_reasoning: false,
                returns_encrypted_reasoning: false,
                supports_tools: true,
            };
        }

        // --- grok-4 family (check specific variants before generic) ---

        // grok-4 multi-agent: encrypted reasoning + effort control
        if name.starts_with("grok-4") && name.contains("multi-agent") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 2.00,
                cached_input_price: 0.50,
                output_price: 6.00,
                context_window: 2_097_152,
                effective_ceiling: 1_200_000,
                supports_reasoning_effort_control: true,
                returns_plaintext_reasoning: false,
                returns_encrypted_reasoning: true,
                supports_tools: true,
            };
        }

        // grok-4 non-reasoning: no reasoning capabilities
        if name.starts_with("grok-4") && name.contains("non-reasoning") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 2.00,
                cached_input_price: 0.50,
                output_price: 6.00,
                context_window: 2_097_152,
                effective_ceiling: 1_200_000,
                supports_reasoning_effort_control: false,
                returns_plaintext_reasoning: false,
                returns_encrypted_reasoning: false,
                supports_tools: true,
            };
        }

        // grok-4 reasoning (default grok-4 family): encrypted reasoning
        if name.starts_with("grok-4") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 2.00,
                cached_input_price: 0.50,
                output_price: 6.00,
                context_window: 2_097_152,
                effective_ceiling: 1_200_000,
                supports_reasoning_effort_control: false,
                returns_plaintext_reasoning: false,
                returns_encrypted_reasoning: true,
                supports_tools: true,
            };
        }

        // --- grok-2 family ---
        if name.starts_with("grok-2") {
            return ModelProfile {
                name: name.to_string(),
                input_price: 2.00,
                cached_input_price: 0.50,
                output_price: 10.00,
                context_window: 131_072,
                effective_ceiling: 80_000,
                supports_reasoning_effort_control: false,
                returns_plaintext_reasoning: false,
                returns_encrypted_reasoning: false,
                supports_tools: true,
            };
        }

        // --- Unknown model: conservative fallback ---
        ModelProfile {
            name: name.to_string(),
            input_price: 0.0,
            cached_input_price: 0.0,
            output_price: 0.0,
            context_window: 131_072,
            effective_ceiling: 80_000,
            supports_reasoning_effort_control: false,
            returns_plaintext_reasoning: false,
            returns_encrypted_reasoning: false,
            supports_tools: true,
        }
    }

    /// Whether this model has any reasoning capability (plaintext or encrypted).
    /// Used by Phase 3 reasoning support; tested now, called later.
    #[allow(dead_code)]
    pub fn supports_reasoning(&self) -> bool {
        self.returns_plaintext_reasoning || self.returns_encrypted_reasoning
    }

    /// Compaction threshold: 60% of the effective ceiling.
    /// When estimated tokens exceed this, heuristic compaction fires.
    pub fn compaction_threshold(&self) -> usize {
        self.effective_ceiling * 60 / 100
    }

    /// Estimate cost for a given usage.
    ///
    /// Reasoning tokens are billed at the output token rate and are already
    /// included in `output_tokens` by the API — no separate calculation needed.
    /// When `cached_input_tokens` are available, those tokens are billed at
    /// `cached_input_price` instead of `input_price`.
    pub fn estimate_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        if self.input_price == 0.0 && self.output_price == 0.0 {
            return None;
        }
        Some(
            (input_tokens as f64 * self.input_price + output_tokens as f64 * self.output_price)
                / 1_000_000.0,
        )
    }

    /// Estimate cost using full Usage details (cached tokens, reasoning tokens).
    #[allow(dead_code)] // Will be used by Phase 4 cache display
    pub fn estimate_cost_from_usage(&self, usage: &crate::api::Usage) -> Option<f64> {
        if self.input_price == 0.0 && self.output_price == 0.0 {
            return None;
        }
        let cached = usage.cached_input_tokens.unwrap_or(0);
        let non_cached = usage.input_tokens.saturating_sub(cached);
        let input_cost =
            non_cached as f64 * self.input_price + cached as f64 * self.cached_input_price;
        let output_cost = usage.output_tokens as f64 * self.output_price;
        Some((input_cost + output_cost) / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- grok-3 family ---

    #[test]
    fn grok_3_known_profile() {
        let p = ModelProfile::for_model("grok-3");
        assert_eq!(p.input_price, 3.00);
        assert_eq!(p.output_price, 15.00);
        assert_eq!(p.cached_input_price, 0.50);
        assert_eq!(p.context_window, 131_072);
        assert!(p.effective_ceiling <= p.context_window);
        assert!(p.supports_tools);
        assert!(!p.supports_reasoning());
    }

    #[test]
    fn grok_3_fast_matches_grok_3() {
        let p = ModelProfile::for_model("grok-3-fast");
        assert_eq!(p.input_price, 3.00);
        assert_eq!(p.output_price, 15.00);
        assert!(!p.supports_reasoning_effort_control);
    }

    #[test]
    fn grok_3_mini_known_profile() {
        let p = ModelProfile::for_model("grok-3-mini");
        assert_eq!(p.input_price, 0.30);
        assert_eq!(p.output_price, 0.50);
        assert_eq!(p.cached_input_price, 0.05);
        assert_eq!(p.context_window, 131_072);
        assert!(p.supports_reasoning_effort_control);
        assert!(p.returns_plaintext_reasoning);
        assert!(!p.returns_encrypted_reasoning);
        assert!(p.supports_tools);
    }

    #[test]
    fn grok_3_mini_fast_matches_mini() {
        let p = ModelProfile::for_model("grok-3-mini-fast");
        assert_eq!(p.input_price, 0.30);
        assert!(p.supports_reasoning_effort_control);
        assert!(p.returns_plaintext_reasoning);
    }

    // --- grok-4 reasoning family ---

    #[test]
    fn grok_4_reasoning_profile() {
        let p = ModelProfile::for_model("grok-4-1-fast-reasoning");
        assert_eq!(p.input_price, 2.00);
        assert_eq!(p.output_price, 6.00);
        assert_eq!(p.cached_input_price, 0.50);
        assert_eq!(p.context_window, 2_097_152);
        assert!(p.effective_ceiling <= p.context_window);
        assert!(!p.supports_reasoning_effort_control);
        assert!(!p.returns_plaintext_reasoning);
        assert!(p.returns_encrypted_reasoning);
        assert!(p.supports_tools);
    }

    #[test]
    fn grok_4_reasoning_date_suffix() {
        let p = ModelProfile::for_model("grok-4-1-fast-reasoning-0415");
        assert_eq!(p.input_price, 2.00);
        assert!(p.returns_encrypted_reasoning);
        assert!(p.supports_tools);
    }

    #[test]
    fn grok_4_base_reasoning() {
        // "grok-4" without qualifiers falls into reasoning family
        let p = ModelProfile::for_model("grok-4");
        assert!(p.returns_encrypted_reasoning);
        assert!(!p.supports_reasoning_effort_control);
    }

    // --- grok-4 non-reasoning family ---

    #[test]
    fn grok_4_non_reasoning_profile() {
        let p = ModelProfile::for_model("grok-4-1-non-reasoning");
        assert_eq!(p.input_price, 2.00);
        assert_eq!(p.output_price, 6.00);
        assert_eq!(p.context_window, 2_097_152);
        assert!(!p.supports_reasoning_effort_control);
        assert!(!p.returns_plaintext_reasoning);
        assert!(!p.returns_encrypted_reasoning);
        assert!(p.supports_tools);
    }

    #[test]
    fn grok_4_non_reasoning_date_suffix() {
        let p = ModelProfile::for_model("grok-4-1-non-reasoning-0309");
        assert!(!p.returns_encrypted_reasoning);
        assert!(!p.supports_reasoning());
    }

    // --- grok-4 multi-agent family ---

    #[test]
    fn grok_4_multi_agent_profile() {
        let p = ModelProfile::for_model("grok-4.20-multi-agent");
        assert_eq!(p.input_price, 2.00);
        assert_eq!(p.output_price, 6.00);
        assert_eq!(p.context_window, 2_097_152);
        assert!(p.supports_reasoning_effort_control);
        assert!(!p.returns_plaintext_reasoning);
        assert!(p.returns_encrypted_reasoning);
        assert!(p.supports_tools);
    }

    #[test]
    fn grok_4_multi_agent_date_suffix() {
        let p = ModelProfile::for_model("grok-4.20-multi-agent-0415");
        assert!(p.supports_reasoning_effort_control);
        assert!(p.returns_encrypted_reasoning);
    }

    // --- grok-2 family ---

    #[test]
    fn grok_2_known_profile() {
        let p = ModelProfile::for_model("grok-2");
        assert_eq!(p.input_price, 2.00);
        assert_eq!(p.output_price, 10.00);
        assert!(p.supports_tools);
        assert!(!p.supports_reasoning());
    }

    // --- Unknown models ---

    #[test]
    fn unknown_model_returns_defaults() {
        let p = ModelProfile::for_model("some-future-model");
        assert_eq!(p.input_price, 0.0);
        assert_eq!(p.output_price, 0.0);
        assert_eq!(p.cached_input_price, 0.0);
        assert!(p.context_window > 0);
        assert!(p.effective_ceiling > 0);
        assert!(p.effective_ceiling <= p.context_window);
        assert!(!p.supports_reasoning());
        assert!(p.supports_tools);
    }

    // --- Cost estimation ---

    #[test]
    fn estimate_cost_known_model() {
        let p = ModelProfile::for_model("grok-3");
        let cost = p.estimate_cost(1_000, 500);
        assert!(cost.is_some());
        let c = cost.unwrap();
        // 1000 * 3.0 / 1M + 500 * 15.0 / 1M = 0.003 + 0.0075 = 0.0105
        assert!((c - 0.0105).abs() < 1e-6);
    }

    #[test]
    fn estimate_cost_unknown_model_returns_none() {
        let p = ModelProfile::for_model("unknown");
        assert!(p.estimate_cost(1000, 500).is_none());
    }

    // --- Compaction thresholds ---

    #[test]
    fn compaction_threshold_131k_models() {
        let p = ModelProfile::for_model("grok-3");
        assert_eq!(p.compaction_threshold(), p.effective_ceiling * 60 / 100);
        // 80_000 * 60 / 100 = 48_000
        assert_eq!(p.compaction_threshold(), 48_000);
    }

    #[test]
    fn compaction_threshold_2m_models() {
        let p = ModelProfile::for_model("grok-4-1-fast-reasoning");
        assert_eq!(p.compaction_threshold(), p.effective_ceiling * 60 / 100);
        // 1_200_000 * 60 / 100 = 720_000
        assert_eq!(p.compaction_threshold(), 720_000);
    }

    #[test]
    fn effective_ceiling_below_context_window() {
        for name in &[
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-2",
            "grok-4-1-fast-reasoning",
            "grok-4-1-non-reasoning",
            "grok-4.20-multi-agent",
            "unknown",
        ] {
            let p = ModelProfile::for_model(name);
            assert!(
                p.effective_ceiling <= p.context_window,
                "effective_ceiling should not exceed context_window for {name}"
            );
        }
    }

    // --- Capability flag accuracy ---

    #[test]
    fn supports_tools_all_known_families() {
        let families = [
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
            "grok-2",
            "grok-4-1-fast-reasoning",
            "grok-4-1-non-reasoning",
            "grok-4.20-multi-agent",
        ];
        for name in families {
            let p = ModelProfile::for_model(name);
            assert!(p.supports_tools, "{name} should support tools");
        }
    }

    #[test]
    fn reasoning_effort_only_on_supported_models() {
        // grok-3-mini and grok-4 multi-agent support effort control
        assert!(ModelProfile::for_model("grok-3-mini").supports_reasoning_effort_control);
        assert!(ModelProfile::for_model("grok-4.20-multi-agent").supports_reasoning_effort_control);

        // Others do not
        assert!(!ModelProfile::for_model("grok-3").supports_reasoning_effort_control);
        assert!(!ModelProfile::for_model("grok-3-fast").supports_reasoning_effort_control);
        assert!(
            !ModelProfile::for_model("grok-4-1-fast-reasoning").supports_reasoning_effort_control
        );
        assert!(
            !ModelProfile::for_model("grok-4-1-non-reasoning").supports_reasoning_effort_control
        );
    }
}
