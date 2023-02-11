use vek::{
    geom::repr_simd::Ray,
    vec::repr_simd::{Rgba, Vec3},
};

use crate::RayResult;

pub enum ObjectKind {
    Sphere { radius: f32, center: Vec3<f32> },
}

#[derive(Clone, Copy)]
pub struct Material {
    pub color: Rgba<f32>,
    pub smoothness: f32,
    pub refraction_index: f32,
}

pub struct Object {
    pub kind: ObjectKind,
    pub material: Material,
}

impl Object {
    pub fn trace(&self, ray: &Ray<f32>) -> Option<RayResult> {
        let (t, pos, norm, inside) = match self.kind {
            ObjectKind::Sphere { center, radius } => {
                let radius2 = radius * radius;
                let l = center - ray.origin;
                let tca = l.dot(ray.direction);
                let d2 = l.magnitude_squared() - tca * tca;
                if d2 > radius2 {
                    return None;
                }
                let thc = (radius2 - d2).sqrt();
                let mut t0 = tca - thc;
                let mut t1 = tca + thc;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                if t0 <= 0.0 {
                    if t1 <= 0.0 {
                        return None;
                    }
                    let pos = ray.origin + ray.direction * t1;

                    (t1, pos, -(pos - center) / radius, true)
                } else {
                    let pos = ray.origin + ray.direction * t0;

                    (t0, pos, (pos - center) / radius, false)
                }
            }
        };

        Some(RayResult { dist: t, pos, norm, inside })
    }
}
