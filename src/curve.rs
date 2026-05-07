//! Animated parameter curves and the matching driver system.
//!
//! Generators are static-config — `BarkConfig`, `MetalConfig`, etc. carry
//! plain numeric fields with no notion of time.  This module adds a
//! lightweight time axis on top: callers describe how a [`TextureConfig`]
//! value should evolve over `t` (seconds since spawn) via a closure, attach
//! that closure to a material entity through
//! [`AnimatedProceduralMaterial`], and the
//! [`tick_animated_procedural_materials`] system re-enqueues generation
//! whenever the closure's output changes.
//!
//! # Throttling
//!
//! Re-running every generator every frame would saturate the rayon pool.
//! Two thresholds gate regeneration:
//!
//! 1. [`AnimatedProceduralMaterial::min_regen_interval`] — wall-clock cooldown
//!    between regeneration attempts.  Default `0.25 s`.
//! 2. [`TextureConfig::fingerprint`] equality — even after the cooldown
//!    elapses, a regeneration is skipped if the closure produced the same
//!    config as last time.  This makes piecewise-constant curves
//!    (e.g. [`Stepped`]) cheap.
//!
//! Clients that want to interpolate visually between regenerations should
//! drive a fragment-shader uniform on the material instead — generator
//! output is RGBA8 pixels and is the wrong knob for sub-second smoothness.
//!
//! # Curves vs closures
//!
//! Concrete [`ParameterCurve`] impls ([`Linear`], [`EaseInOut`], [`Stepped`],
//! [`ScriptedFn`]) cover the common shapes for individual numeric fields.
//! Bind a curve into a [`TextureConfig`] by calling `.eval(t)` inside the
//! `texture_curve` closure:
//!
//! ```rust,ignore
//! let rust = Linear { from: 0.0, to: 1.0, duration: 10.0 };
//! AnimatedProceduralMaterial::new(material, 512, 512, move |t| {
//!     TextureConfig::Metal(MetalConfig {
//!         rust_level: rust.eval(t) as f64,
//!         ..base.clone()
//!     })
//! })
//! ```

use std::sync::Arc;

use bevy::ecs::component::Component;
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Commands, Query};
use bevy::pbr::StandardMaterial;
use bevy::prelude::{Handle, Res};
use bevy::time::Time;

use crate::cache::TextureCacheKey;
use crate::material::{PatchMaterialTextures, TextureConfig};

/// A function from a normalised or absolute time `t` to a value of type `T`.
///
/// All implementations are pure, deterministic, and `Send + Sync`.  Users
/// typically don't implement this directly — they construct one of the
/// concrete curves below and call [`eval`](ParameterCurve::eval) inside the
/// `texture_curve` closure of an [`AnimatedProceduralMaterial`].
pub trait ParameterCurve<T>: Send + Sync {
    /// Sample the curve at time `t` (seconds, monotonically increasing).
    ///
    /// Implementations must clamp out-of-range inputs themselves — the
    /// driver does not pre-process `t`.
    fn eval(&self, t: f32) -> T;
}

/// Linear interpolation from `from` at `t = 0` to `to` at `t = duration`.
/// Holds at `to` for `t >= duration`.
#[derive(Clone, Copy, Debug)]
pub struct Linear<T> {
    pub from: T,
    pub to: T,
    pub duration: f32,
}

/// Smoothstep interpolation: `3·u² − 2·u³` where `u = t / duration`.
/// Useful when a parameter should glide rather than ramp linearly.
#[derive(Clone, Copy, Debug)]
pub struct EaseInOut<T> {
    pub from: T,
    pub to: T,
    pub duration: f32,
}

/// Piecewise-constant curve: holds each `(start_t, value)` until the next
/// step's start.  Steps must be sorted by `start_t` ascending; out-of-range
/// `t` clamps to the nearest step.
#[derive(Clone, Debug)]
pub struct Stepped<T: Clone> {
    /// `(start_t, value)` pairs; sorted ascending by `start_t`.
    pub steps: Vec<(f32, T)>,
}

/// Wraps an arbitrary closure as a [`ParameterCurve`].  Use when none of the
/// canned curves fit (sine waves, randomised wobbles, externally-driven
/// data).
pub struct ScriptedFn<T, F>(pub F)
where
    F: Fn(f32) -> T + Send + Sync;

// --- impls ------------------------------------------------------------------

impl ParameterCurve<f32> for Linear<f32> {
    fn eval(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return self.to;
        }
        let u = (t / self.duration).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * u
    }
}

impl ParameterCurve<f64> for Linear<f64> {
    fn eval(&self, t: f32) -> f64 {
        if self.duration <= 0.0 {
            return self.to;
        }
        let u = (t / self.duration).clamp(0.0, 1.0) as f64;
        self.from + (self.to - self.from) * u
    }
}

impl ParameterCurve<f32> for EaseInOut<f32> {
    fn eval(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return self.to;
        }
        let u = (t / self.duration).clamp(0.0, 1.0);
        let s = u * u * (3.0 - 2.0 * u);
        self.from + (self.to - self.from) * s
    }
}

impl ParameterCurve<f64> for EaseInOut<f64> {
    fn eval(&self, t: f32) -> f64 {
        if self.duration <= 0.0 {
            return self.to;
        }
        let u = (t / self.duration).clamp(0.0, 1.0) as f64;
        let s = u * u * (3.0 - 2.0 * u);
        self.from + (self.to - self.from) * s
    }
}

impl<T: Clone + Send + Sync> ParameterCurve<T> for Stepped<T> {
    fn eval(&self, t: f32) -> T {
        debug_assert!(
            !self.steps.is_empty(),
            "Stepped curve requires at least one step"
        );
        // partition_point finds the first step whose start_t > t; the active
        // step is the one immediately before it.  When t precedes every
        // step we still clamp to the first.
        let idx = self
            .steps
            .partition_point(|(s, _)| *s <= t)
            .saturating_sub(1);
        self.steps[idx].1.clone()
    }
}

impl<T, F> ParameterCurve<T> for ScriptedFn<T, F>
where
    F: Fn(f32) -> T + Send + Sync,
{
    fn eval(&self, t: f32) -> T {
        (self.0)(t)
    }
}

// --- driver -----------------------------------------------------------------

/// Type-erased closure producing a [`TextureConfig`] from elapsed time.
/// Stored inside [`AnimatedProceduralMaterial`].
pub type TextureCurve = Arc<dyn Fn(f32) -> TextureConfig + Send + Sync>;

/// Component that drives a procedural-texture refresh on a target
/// [`StandardMaterial`] as time advances.
///
/// Spawned alongside the material; consumed by
/// [`tick_animated_procedural_materials`], which re-runs `texture_curve`
/// each tick and dispatches a new generation when the result changes.
#[derive(Component)]
pub struct AnimatedProceduralMaterial {
    /// Material whose texture slots will be patched.
    pub material: Handle<StandardMaterial>,
    /// Texture resolution for every regeneration.
    pub width: u32,
    pub height: u32,
    /// Closure: `t` (seconds since spawn) -> next `TextureConfig`.
    pub texture_curve: TextureCurve,
    /// Minimum delay between regeneration attempts.  Lower = smoother but
    /// more CPU; higher = jumpier but cheap.  Default `0.25 s`.
    pub min_regen_interval: f32,
    /// Internal: total elapsed time since this component was spawned.
    pub elapsed: f32,
    /// Internal: time of the last regeneration attempt (success or skip).
    pub last_regen_at: f32,
    /// Internal: fingerprint of the most recently dispatched config.
    /// Used to skip redundant regenerations when the curve plateaued.
    pub last_fingerprint: u64,
}

impl AnimatedProceduralMaterial {
    /// Default `min_regen_interval` — quarter-second cadence balances
    /// visible motion against pool saturation at typical 256–1024 px sizes.
    pub const DEFAULT_REGEN_INTERVAL: f32 = 0.25;

    /// Build a fresh animator.  The `texture_curve` closure is sampled at
    /// `t = 0` on the first tick; pick its initial value to match whatever
    /// the static material was created with so the first regeneration is
    /// a true delta.
    pub fn new(
        material: Handle<StandardMaterial>,
        width: u32,
        height: u32,
        texture_curve: impl Fn(f32) -> TextureConfig + Send + Sync + 'static,
    ) -> Self {
        Self {
            material,
            width,
            height,
            texture_curve: Arc::new(texture_curve),
            min_regen_interval: Self::DEFAULT_REGEN_INTERVAL,
            elapsed: 0.0,
            last_regen_at: f32::NEG_INFINITY,
            last_fingerprint: 0,
        }
    }

    /// Sets [`min_regen_interval`](Self::min_regen_interval) and returns
    /// `self` for chaining.
    pub fn with_min_regen_interval(mut self, interval: f32) -> Self {
        self.min_regen_interval = interval.max(0.0);
        self
    }
}

/// Bevy system — advances every [`AnimatedProceduralMaterial`]'s clock,
/// re-evaluates its curve, and dispatches a generation task whenever the
/// fingerprint of the next config differs from the previous one.
///
/// Registered automatically by
/// [`SymbiosTexturePlugin`](crate::SymbiosTexturePlugin).
pub fn tick_animated_procedural_materials(
    time: Res<Time>,
    mut commands: Commands,
    mut anim_q: Query<(Entity, &mut AnimatedProceduralMaterial)>,
) {
    let dt = time.delta_secs();

    for (_entity, mut anim) in &mut anim_q {
        anim.elapsed += dt;
        let since = anim.elapsed - anim.last_regen_at;
        if since < anim.min_regen_interval {
            continue;
        }

        let cfg = (anim.texture_curve)(anim.elapsed);
        let fp = cfg.fingerprint();
        anim.last_regen_at = anim.elapsed;

        // Curve plateau — fingerprint unchanged means no need to regenerate.
        if fp == anim.last_fingerprint {
            continue;
        }
        anim.last_fingerprint = fp;

        let key = TextureCacheKey {
            kind: cfg.label(),
            fingerprint: fp,
            width: anim.width,
            height: anim.height,
        };

        if let Some(pending) = cfg.spawn(anim.width, anim.height) {
            commands.spawn((
                pending,
                PatchMaterialTextures {
                    target: anim.material.clone(),
                    cache_key: Some(key),
                },
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear endpoints + midpoint round-trip.
    #[test]
    fn linear_clamps_and_interpolates() {
        let c = Linear {
            from: 0.0_f32,
            to: 4.0,
            duration: 2.0,
        };
        assert_eq!(c.eval(0.0), 0.0);
        assert_eq!(c.eval(2.0), 4.0);
        assert!((c.eval(1.0) - 2.0).abs() < 1e-6);
        // Out-of-range clamps to the nearest endpoint.
        assert_eq!(c.eval(-5.0), 0.0);
        assert_eq!(c.eval(99.0), 4.0);
    }

    /// EaseInOut hits both endpoints exactly and is symmetric around midpoint.
    #[test]
    fn ease_in_out_hits_endpoints_and_midpoint() {
        let c = EaseInOut {
            from: 0.0_f32,
            to: 1.0,
            duration: 1.0,
        };
        assert_eq!(c.eval(0.0), 0.0);
        assert_eq!(c.eval(1.0), 1.0);
        assert!((c.eval(0.5) - 0.5).abs() < 1e-6);
    }

    /// Stepped picks the most-recent step whose `start_t` <= `t`.
    #[test]
    fn stepped_holds_until_next_step() {
        let c = Stepped {
            steps: vec![(0.0, 0_i32), (1.0, 1), (3.0, 2), (5.0, 3)],
        };
        assert_eq!(c.eval(0.0), 0);
        assert_eq!(c.eval(0.99), 0);
        assert_eq!(c.eval(1.0), 1);
        assert_eq!(c.eval(2.5), 1);
        assert_eq!(c.eval(3.0), 2);
        assert_eq!(c.eval(100.0), 3);
    }

    /// ScriptedFn forwards verbatim.
    #[test]
    fn scripted_fn_forwards() {
        let c = ScriptedFn(|t: f32| t * 2.0 + 1.0);
        assert_eq!(c.eval(0.0), 1.0);
        assert_eq!(c.eval(3.0), 7.0);
    }
}
