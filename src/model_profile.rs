/// Model metadata: pricing, context limits, and effective ceilings.
///
/// Used by the cost estimator now and by compaction thresholds in later phases.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    pub name: String,
    /// Input price per 1M tokens (USD)
    pub input_price: f64,
    /// Output price per 1M tokens (USD)
    pub output_price: f64,
    /// Raw context window in tokens
    pub context_window: usize,
    /// Conservative "smart zone" cap — compaction threshold is 60% of this
    pub effective_ceiling: usize,
}

impl ModelProfile {
    /// Look up a model by name. Returns known profiles for grok models,
    /// sensible defaults for anything else.
    pub fn for_model(name: &str) -> ModelProfile {
        match name {
            n if n.starts_with("grok-3-mini") => ModelProfile {
                name: name.to_string(),
                input_price: 0.30,
                output_price: 0.50,
                context_window: 131_072,
                effective_ceiling: 80_000,
            },
            n if n.starts_with("grok-3") => ModelProfile {
                name: name.to_string(),
                input_price: 3.00,
                output_price: 15.00,
                context_window: 131_072,
                effective_ceiling: 80_000,
            },
            n if n.starts_with("grok-2") => ModelProfile {
                name: name.to_string(),
                input_price: 2.00,
                output_price: 10.00,
                context_window: 131_072,
                effective_ceiling: 80_000,
            },
            _ => ModelProfile {
                name: name.to_string(),
                input_price: 0.0,
                output_price: 0.0,
                context_window: 131_072,
                effective_ceiling: 80_000,
            },
        }
    }

    /// Compaction threshold: 60% of the effective ceiling.
    /// When estimated tokens exceed this, heuristic compaction fires.
    pub fn compaction_threshold(&self) -> usize {
        self.effective_ceiling * 60 / 100
    }

    /// Estimate cost for a given usage.
    pub fn estimate_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        if self.input_price == 0.0 && self.output_price == 0.0 {
            return None;
        }
        Some(
            (input_tokens as f64 * self.input_price + output_tokens as f64 * self.output_price)
                / 1_000_000.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grok_3_known_profile() {
        let p = ModelProfile::for_model("grok-3");
        assert_eq!(p.input_price, 3.00);
        assert_eq!(p.output_price, 15.00);
        assert_eq!(p.context_window, 131_072);
        assert!(p.effective_ceiling <= p.context_window);
    }

    #[test]
    fn grok_3_fast_matches_grok_3() {
        let p = ModelProfile::for_model("grok-3-fast");
        assert_eq!(p.input_price, 3.00);
        assert_eq!(p.output_price, 15.00);
    }

    #[test]
    fn grok_3_mini_known_profile() {
        let p = ModelProfile::for_model("grok-3-mini");
        assert_eq!(p.input_price, 0.30);
        assert_eq!(p.output_price, 0.50);
        assert_eq!(p.context_window, 131_072);
    }

    #[test]
    fn grok_3_mini_fast_matches_mini() {
        let p = ModelProfile::for_model("grok-3-mini-fast");
        assert_eq!(p.input_price, 0.30);
    }

    #[test]
    fn grok_2_known_profile() {
        let p = ModelProfile::for_model("grok-2");
        assert_eq!(p.input_price, 2.00);
        assert_eq!(p.output_price, 10.00);
    }

    #[test]
    fn unknown_model_returns_defaults() {
        let p = ModelProfile::for_model("some-future-model");
        assert_eq!(p.input_price, 0.0);
        assert_eq!(p.output_price, 0.0);
        assert!(p.context_window > 0);
        assert!(p.effective_ceiling > 0);
        assert!(p.effective_ceiling <= p.context_window);
    }

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

    #[test]
    fn compaction_threshold_is_60_percent_of_ceiling() {
        let p = ModelProfile::for_model("grok-3");
        assert_eq!(p.compaction_threshold(), p.effective_ceiling * 60 / 100);
        // 80_000 * 60 / 100 = 48_000
        assert_eq!(p.compaction_threshold(), 48_000);
    }

    #[test]
    fn effective_ceiling_below_context_window() {
        for name in &["grok-3", "grok-3-fast", "grok-3-mini", "grok-2", "unknown"] {
            let p = ModelProfile::for_model(name);
            assert!(
                p.effective_ceiling <= p.context_window,
                "effective_ceiling should not exceed context_window for {name}"
            );
        }
    }
}
