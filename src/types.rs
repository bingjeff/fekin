//! Shared linear-algebra type aliases and naming conventions.
//!
//! Type alias naming:
//! - `<Base><Size><Scalar>` for fixed sizes (for example, `Vector3d`, `Matrix4f`).
//! - `<Base><Scalar><...>` for generic forms (for example, `Vectorf<N>`, `Pointd<N>`).
//! - Scalar suffix: `f` => `f32`, `d` => `f64`.
//!
//! Frame-aware variable naming:
//! - Transforms use `<reference>_tfm_<frame>` and compose as `a_tfm_b * b_tfm_c`.
//! - Points use `<from>_xyz_<to>` and compose as `a_xyz_c = a_tfm_b * b_xyz_c`.
//! - Left postfix should match right prefix during composition.
//!
//! Geometric meaning:
//! - A point is a fixed vector (position): an isometry rotates and translates it.
//! - A vector is a free vector (direction/displacement): an isometry only rotates it.
//!
use nalgebra as na;

// Free vectors: rotation acts, translation does not.
pub type Vectorf<const N: usize> = na::SVector<f32, N>;
pub type Vectord<const N: usize> = na::SVector<f64, N>;

pub type Matrixf<const R: usize, const C: usize> = na::SMatrix<f32, R, C>;
pub type Matrixd<const R: usize, const C: usize> = na::SMatrix<f64, R, C>;

// Fixed vectors (points): isometries apply both rotation and translation.
pub type Pointf<const N: usize> = na::OPoint<f32, na::Const<N>>;
pub type Pointd<const N: usize> = na::OPoint<f64, na::Const<N>>;

pub type Vector3f = Vectorf<3>;
pub type Vector3d = Vectord<3>;
pub type Vector4f = Vectorf<4>;
pub type Vector4d = Vectord<4>;
pub type Vector6f = Vectorf<6>;
pub type Vector6d = Vectord<6>;

pub type Point3f = Pointf<3>;
pub type Point3d = Pointd<3>;

pub type Matrix3f = Matrixf<3, 3>;
pub type Matrix3d = Matrixd<3, 3>;
pub type Matrix4f = Matrixf<4, 4>;
pub type Matrix4d = Matrixd<4, 4>;
pub type Matrix6f = Matrixf<6, 6>;
pub type Matrix6d = Matrixd<6, 6>;

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
    fn vectorf_supports_generic_dimension() {
        let a_xyz_b: Vectorf<3> = Vectorf::from_row_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(a_xyz_b.nrows(), 3);
        assert_eq!(a_xyz_b.ncols(), 1);
    }

    #[test]
    fn point3d_has_expected_shape() {
        let a_xyz_b = Point3d::new(1.0, 2.0, 3.0);
        assert_eq!(a_xyz_b.coords.nrows(), 3);
        assert_eq!(a_xyz_b.coords.ncols(), 1);
    }

    #[test]
    fn pointf_supports_generic_dimension() {
        let a_xyz_b: Pointf<3> = na::Point3::new(1.0, 2.0, 3.0);
        assert_eq!(a_xyz_b.coords.nrows(), 3);
        assert_eq!(a_xyz_b.coords.ncols(), 1);
    }

    #[test]
    fn matrix4d_identity_is_square() {
        let a_tfm_b = Matrix4d::identity();
        assert_eq!(a_tfm_b.nrows(), 4);
        assert_eq!(a_tfm_b.ncols(), 4);
    }

    #[test]
    fn matrixf_supports_generic_row_and_column_sizes() {
        let a_mat_b: Matrixf<2, 3> = Matrixf::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a_mat_b.nrows(), 2);
        assert_eq!(a_mat_b.ncols(), 3);
    }

    #[test]
    fn transform3d_applies_to_point() {
        let a_tfm_b = Transform3d::translation(1.0, -2.0, 0.5);
        let b_xyz_c = Point3d::new(2.0, 3.0, 4.0);
        let a_xyz_c = a_tfm_b * b_xyz_c;
        assert_eq!(a_xyz_c, Point3d::new(3.0, 1.0, 4.5));
    }
}
