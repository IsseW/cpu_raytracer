mod object;

use std::time::Instant;

use object::{Material, Object, ObjectKind};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use vek::{
    geom::repr_simd::Ray,
    quaternion::repr_simd::Quaternion,
    transform::repr_simd::Transform,
    vec::repr_simd::{Rgb, Rgba, Vec2, Vec3},
};

pub struct RayResult {
    dist: f32,
    pos: Vec3<f32>,
    norm: Vec3<f32>,
    inside: bool,
}

struct Camera {
    pub transform: Transform<f32, f32, f32>,
    pub zoom: f32,
}

#[derive(Copy, Clone)]
struct DirLight {
    direction: Vec3<f32>,
    color: Rgb<f32>,
}

struct World {
    objects: Vec<Object>,
    dir_lights: Vec<DirLight>,
    camera: Camera,
}

#[inline(always)]
fn unit_lerp(a: Vec3<f32>, b: Vec3<f32>, t0: f32, t1: f32) -> Vec3<f32> {
    (a * t0 + b * t1).normalized()
}

/// https://observablehq.com/@mourner/approximating-geometric-slerp
#[inline(always)]
fn fslerp(a: Vec3<f32>, b: Vec3<f32>, t: f32) -> Vec3<f32> {
    if t == 0.5 {
        return unit_lerp(a, b, t, t);
    };
    let d = a.dot(b);
    if d < 0.0 {
        // angle > 90; recurse into one of the halves
        let m = unit_lerp(a, b, 0.5, 0.5);
        return if t < 0.5 {
            fslerp(a, m, t * 2.0)
        } else {
            fslerp(m, b, t * 2.0 - 1.0)
        };
    }
    let aa = 1.0904 + d * (-3.2452 + d * (3.55645 - d * 1.43519));
    let bb = 0.848013 + d * (-1.06021 + d * 0.215638);
    let k = aa * (t - 0.5) * (t - 0.5) + bb;
    let p = t + t * (t - 0.5) * (t - 1.0) * k;
    unit_lerp(a, b, 1.0 - p, p)
}

#[derive(Clone, Copy)]
struct Medium {
    refraction_index: f32,
    color: Rgba<f32>,
}

impl Default for Medium {
    fn default() -> Self {
        Self { refraction_index: 1.0, color: Rgba::new(1.0, 1.0, 1.0, 0.0) }
    }
}

impl World {
    fn render_sky(&self, _dir: Vec3<f32>) -> Rgb<f32> {
        return Rgb::new(0.3, 0.3, 0.3);
    }

    fn trace_ray(&self, ray: Ray<f32>) -> Option<(&Object, RayResult)> {
        self.objects
            .iter()
            .filter_map(|obj| Some((obj, obj.trace(&ray)?)))
            .min_by(|a, b| {
                a.1.dist
                    .partial_cmp(&b.1.dist)
                    .unwrap_or(std::cmp::Ordering::Less)
            })
    }

    fn render_ray(&self, ray: Ray<f32>, depth: u32, medium: Medium, rng: &mut impl Rng) -> Rgb<f32> {
        self.trace_ray(ray)
            .map(|(obj, ray_result)| {
                let medium_travel_color = medium.color.rgb() * (1.0 - (1.0 - medium.color.a).powf(ray_result.dist));
                let (light_color, specular_color) = self
                    .dir_lights
                    .iter()
                    .filter(|light| {
                        light.direction.dot(ray_result.norm) > 0.0 ||
                        self.trace_ray(Ray {
                            origin: ray_result.pos - light.direction * 0.01,
                            direction: -light.direction,
                        })
                        .is_none()
                    })
                    .map(|light| {
                        let diffuse = (-light.direction)
                            .dot(ray_result.norm)
                            .max(0.0);

                        let r = light.direction.reflected(ray_result.norm);
                        let specular = r.dot(ray.direction).powf(100.0) * (1.0 - obj.material.smoothness);

                        (light.color * diffuse, specular * light.color)
                    })
                    .reduce(|a, b| (a.0 + b.0, a.1 + b.1)).unwrap_or_default();

                let color = if depth > 0 {
                    let direction = ray.direction.reflected(ray_result.norm);

                    let random_dir = {
                        let dir = Vec3::broadcast(())
                            .map(|_| rng.gen_range(-1.0..1.0))
                            .try_normalized()
                            .unwrap_or(ray_result.norm);
                        if dir.dot(ray_result.norm) < 0.0 {
                            dir.reflected(ray_result.norm)
                        } else {
                            dir
                        }
                    };

                    let direction = fslerp(direction, random_dir, obj.material.smoothness);

                    let ray = Ray {
                        origin: ray_result.pos + direction * 0.01,
                        direction,
                    };
                    let ray_color = self.render_ray(ray, depth - 1, medium, rng) * 0.8;

                    let r = (-direction).reflected(ray_result.norm);
                    let specular = r.dot(ray.direction).powf(50.0) * (1.0 - obj.material.smoothness);

                    let surface = (ray_color + light_color) * obj.material.color.rgb() + specular_color + specular;

                    let direction = ray.direction.refracted(ray_result.norm, obj.material.refraction_index / medium.refraction_index);
                    if obj.material.color.a < 1.0 && ray_result.norm.dot(direction) < 0.0 {
                        let direction = fslerp(direction, -random_dir, obj.material.smoothness);
                        let ray = Ray {
                            origin: ray_result.pos + direction * 0.01,
                            direction,
                        };

                        let ray_color = self.render_ray(ray, depth - 1, Medium {
                            refraction_index: obj.material.refraction_index,
                            color: obj.material.color,
                        }, rng);

                        surface * obj.material.color.a + ray_color * (1.0 - obj.material.color.a)
                    } else {
                        surface
                    }
                } else {
                    obj.material.color.rgb() * light_color
                };

                color * medium.color.rgb() + medium_travel_color
            })
            .unwrap_or_else(|| self.render_sky(ray.direction))
    }

    fn render(&self, size: Vec2<u32>, samples: u32, ray_depth: u32) -> Vec<Rgb<f32>> {
        (0..size.y)
            .into_par_iter()
            .flat_map(|y| {
                let mut rng = SmallRng::seed_from_u64(y as u64);
                (0..size.x)
                    .map(|x| {
                        let (ray, right, up) = self.get_ray(Vec2::new(x, y), size);
                        (0..samples)
                            .map(|_| {
                                let ray = Ray {
                                    origin: ray.origin
                                        + right * rng.gen_range(-1.0..1.0)
                                        + up * rng.gen_range(-1.0..1.0),
                                    direction: ray.direction,
                                };
                                self.render_ray(ray, ray_depth, Medium::default(), &mut rng)
                            })
                            .sum::<Rgb<f32>>()
                            / samples as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn get_ray(&self, pixel: Vec2<u32>, size: Vec2<u32>) -> (Ray<f32>, Vec3<f32>, Vec3<f32>) {
        let origin = self.camera.transform.position;

        let forward = self.camera.transform.orientation * Vec3::unit_y();
        let right = self.camera.transform.orientation * Vec3::unit_x();
        let up = self.camera.transform.orientation * Vec3::unit_z();

        let cl = ((pixel.as_::<f32>() - (size.as_::<f32>() + 1.0) / 2.0) / size.y as f32) * 2.0;
        let look_point = self.camera.zoom * forward + right * cl.x + up * cl.y;

        let direction = look_point.normalized();

        (
            Ray { origin, direction },
            right / size.y as f32,
            up / size.y as f32,
        )
    }
}

fn main() {
    let red = Material {
        color: Rgba::new(1.0, 0.2, 0.2, 1.0),
        smoothness: 0.1,
        refraction_index: 1.52,
    };
    let green = Material {
        color: Rgba::new(0.2, 1.0, 0.2, 0.0),
        smoothness: 0.02,
        refraction_index: 2.52,
    };
    let white = Material {
        color: Rgba::white(),
        smoothness: 0.9,
        refraction_index: 1.52,
    };
    let objects = vec![
        Object {
            kind: ObjectKind::Sphere {
                radius: 1.0,
                center: Vec3::new(0.0, 0.0, 0.0),
            },
            material: green,
        },
        Object {
            kind: ObjectKind::Sphere {
                radius: 0.3,
                center: Vec3::new(0.0, 0.0, 0.0),
            },
            material: red,
        },
        Object {
            kind: ObjectKind::Sphere {
                radius: 999.0,
                center: Vec3::new(0.0, 0.0, 1000.0),
            },
            material: white,
        },
        Object {
            kind: ObjectKind::Sphere {
                radius: 998.0,
                center: Vec3::new(0.0, 1000.0, 0.0),
            },
            material: white,
        },
    ];
    let world = World {
        objects,
        camera: Camera {
            transform: Transform {
                position: Vec3::new(0.0, -5.0, 0.0),
                orientation: Quaternion::identity(),
                scale: Vec3::one(),
            },
            zoom: 4.0,
        },
        dir_lights: vec![DirLight {
            direction: Vec3::new(0.0, 1.0, 0.5).normalized(),
            color: Rgb::new(0.7, 1.0, 1.0),
        }],
    };

    let start = Instant::now();

    let size = Vec2::new(512, 512);
    let data = world.render(size, 100, 16);

    let max = data
        .iter()
        .map(|color| color.reduce_partial_max())
        .reduce(|a, b| a.max(b))
        .unwrap()
        .clamp(1.0, 2.0);

    let data = data
        .into_iter()
        .flat_map(|rgb| {
            let rgb = rgb.map(|f| (f * 255.0 / max) as u8);
            [rgb.r, rgb.g, rgb.b, 255]
        })
        .collect::<Vec<_>>();

    image::save_buffer_with_format(
        "result.png",
        &data,
        size.x,
        size.y,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();

    println!("Finished in {:.2}s", start.elapsed().as_secs_f32());
}
