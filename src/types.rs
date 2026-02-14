use nalgebra as na;

pub type Vector3f = na::SVector<f32, 3>;
pub type Vector3d = na::SVector<f64, 3>;
pub type Vector4f = na::SVector<f32, 4>;
pub type Vector4d = na::SVector<f64, 4>;
pub type Vector6f = na::SVector<f32, 6>;
pub type Vector6d = na::SVector<f64, 6>;

pub type Matrix3f = na::SMatrix<f32, 3, 3>;
pub type Matrix3d = na::SMatrix<f64, 3, 3>;
pub type Matrix4f = na::SMatrix<f32, 4, 4>;
pub type Matrix4d = na::SMatrix<f64, 4, 4>;
pub type Matrix6f = na::SMatrix<f32, 6, 6>;
pub type Matrix6d = na::SMatrix<f64, 6, 6>;

pub type Transform3f = na::Isometry3<f32>;
pub type Transform3d = na::Isometry3<f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    #[test]
    fn vector3d_has_expected_shape() {
        let a_xyz_b = Vector3d::new(1.0, 2.0, 3.0);
        assert_eq!(a_xyz_b.nrows(), 3);
        assert_eq!(a_xyz_b.ncols(), 1);
    }

    #[test]
    fn matrix4d_identity_is_square() {
        let a_tfm_b = Matrix4d::identity();
        assert_eq!(a_tfm_b.nrows(), 4);
        assert_eq!(a_tfm_b.ncols(), 4);
    }

    #[test]
    fn transform3d_applies_to_point() {
        let a_tfm_b = Transform3d::translation(1.0, -2.0, 0.5);
        let b_xyz_c = na::Point3::new(2.0, 3.0, 4.0);
        let a_xyz_c = a_tfm_b * b_xyz_c;
        assert_eq!(a_xyz_c, na::Point3::new(3.0, 1.0, 4.5));
    }
}
