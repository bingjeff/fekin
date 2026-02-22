//! Jeff Bingham
//! 2010.07.05 (original MATLAB version)
//!
//! Frame class based on the paper:
//! Johnson and Murphey. 2009. "Scalable Variational Integrators for
//! Constrained Mechanical Systems in Generalized Coordinates" - IEEE
//! Transactions on Robotics, 25(6) p1249-1261.
//!
//! This module is a direct Rust skeleton translation of the MATLAB `frame`
//! handle class. The core cache fields and update/partial equations are kept
//! close to the original naming and behavior.
//!
//! Rust-specific differences from MATLAB:
//! - Parent/children relationships use `Rc<RefCell<Frame>>` + `Weak`.
//! - Methods that traverse or mutate the tree are associated functions that
//!   take `&FrameRef` instead of MATLAB handle methods.
//! - Frame identity checks (`self == i_frame`) use `Rc::ptr_eq`.

use crate::types::Matrix4d;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub type FrameRef = Rc<RefCell<Frame>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateType {
    #[default]
    None,
    XTran,
    YTran,
    ZTran,
    XRot,
    YRot,
    ZRot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameData {
    // Transformations to local and global coordinates.
    parent_w_this: Matrix4d,
    global_w_this: Matrix4d,

    // First partial with respect to this frame.
    parent_dw_this: Matrix4d,

    // Second partial with respect to this frame.
    parent_ddw_this: Matrix4d,

    // Inverses.
    this_w_parent: Matrix4d,
    this_dw_parent: Matrix4d,
    this_ddw_parent: Matrix4d,

    // Twist matrix
    parent_z_this: Matrix4d,

    // Frame velocity
    global_v_this: Matrix4d,
}

impl Default for FrameData {
    fn default() -> Self {
        Self {
            parent_w_this: Matrix4d::identity(),
            global_w_this: Matrix4d::identity(),
            parent_dw_this: Matrix4d::zeros(),
            parent_ddw_this: Matrix4d::zeros(),
            this_w_parent: Matrix4d::identity(),
            this_dw_parent: Matrix4d::zeros(),
            this_ddw_parent: Matrix4d::zeros(),
            parent_z_this: Matrix4d::zeros(),
            global_v_this: Matrix4d::zeros(),
        }
    }
}

fn update_translation_x(data: &mut FrameData, q: f64) {
    data.parent_w_this[(0, 3)] = q;
    data.this_w_parent[(0, 3)] = -q;
    data.parent_dw_this[(0, 3)] = 1.0;
    data.this_dw_parent[(0, 3)] = -1.0;
    data.parent_z_this[(0, 3)] = 1.0;
}

fn update_translation_y(data: &mut FrameData, q: f64) {
    data.parent_w_this[(1, 3)] = q;
    data.this_w_parent[(1, 3)] = -q;
    data.parent_dw_this[(1, 3)] = 1.0;
    data.this_dw_parent[(1, 3)] = -1.0;
    data.parent_z_this[(1, 3)] = 1.0;
}

fn update_translation_z(data: &mut FrameData, q: f64) {
    data.parent_w_this[(2, 3)] = q;
    data.this_w_parent[(2, 3)] = -q;
    data.parent_dw_this[(2, 3)] = 1.0;
    data.this_dw_parent[(2, 3)] = -1.0;
    data.parent_z_this[(2, 3)] = 1.0;
}

fn update_rotation_x(data: &mut FrameData, q: f64) {
    let cq = q.cos();
    let sq = q.sin();

    data.parent_w_this[(1, 1)] = cq;
    data.parent_w_this[(1, 2)] = -sq;
    data.parent_w_this[(2, 1)] = sq;
    data.parent_w_this[(2, 2)] = cq;

    data.parent_dw_this[(1, 1)] = -sq;
    data.parent_dw_this[(1, 2)] = -cq;
    data.parent_dw_this[(2, 1)] = cq;
    data.parent_dw_this[(2, 2)] = -sq;

    data.parent_ddw_this[(1, 1)] = -cq;
    data.parent_ddw_this[(1, 2)] = sq;
    data.parent_ddw_this[(2, 1)] = -sq;
    data.parent_ddw_this[(2, 2)] = -cq;

    data.this_w_parent[(1, 1)] = cq;
    data.this_w_parent[(1, 2)] = sq;
    data.this_w_parent[(2, 1)] = -sq;
    data.this_w_parent[(2, 2)] = cq;

    data.this_dw_parent[(1, 1)] = -sq;
    data.this_dw_parent[(1, 2)] = cq;
    data.this_dw_parent[(2, 1)] = -cq;
    data.this_dw_parent[(2, 2)] = -sq;

    data.this_ddw_parent[(1, 1)] = -cq;
    data.this_ddw_parent[(1, 2)] = -sq;
    data.this_ddw_parent[(2, 1)] = sq;
    data.this_ddw_parent[(2, 2)] = -cq;

    data.parent_z_this = data.parent_w_this.transpose() * data.parent_dw_this;
}

fn update_rotation_y(data: &mut FrameData, q: f64) {
    let cq = q.cos();
    let sq = q.sin();

    data.parent_w_this[(0, 0)] = cq;
    data.parent_w_this[(0, 2)] = sq;
    data.parent_w_this[(2, 0)] = -sq;
    data.parent_w_this[(2, 2)] = cq;

    data.parent_dw_this[(0, 0)] = -sq;
    data.parent_dw_this[(0, 2)] = cq;
    data.parent_dw_this[(2, 0)] = -cq;
    data.parent_dw_this[(2, 2)] = -sq;

    data.parent_ddw_this[(0, 0)] = -cq;
    data.parent_ddw_this[(0, 2)] = -sq;
    data.parent_ddw_this[(2, 0)] = sq;
    data.parent_ddw_this[(2, 2)] = -cq;

    data.this_w_parent[(0, 0)] = cq;
    data.this_w_parent[(0, 2)] = -sq;
    data.this_w_parent[(2, 0)] = sq;
    data.this_w_parent[(2, 2)] = cq;

    data.this_dw_parent[(0, 0)] = -sq;
    data.this_dw_parent[(0, 2)] = -cq;
    data.this_dw_parent[(2, 0)] = cq;
    data.this_dw_parent[(2, 2)] = -sq;

    data.this_ddw_parent[(0, 0)] = -cq;
    data.this_ddw_parent[(0, 2)] = sq;
    data.this_ddw_parent[(2, 0)] = -sq;
    data.this_ddw_parent[(2, 2)] = -cq;

    data.parent_z_this = data.parent_w_this.transpose() * data.parent_dw_this;
}

fn update_rotation_z(data: &mut FrameData, q: f64) {
    let cq = q.cos();
    let sq = q.sin();

    data.parent_w_this[(0, 0)] = cq;
    data.parent_w_this[(0, 1)] = -sq;
    data.parent_w_this[(1, 0)] = sq;
    data.parent_w_this[(1, 1)] = cq;

    data.parent_dw_this[(0, 0)] = -sq;
    data.parent_dw_this[(0, 1)] = -cq;
    data.parent_dw_this[(1, 0)] = cq;
    data.parent_dw_this[(1, 1)] = -sq;

    data.parent_ddw_this[(0, 0)] = -cq;
    data.parent_ddw_this[(0, 1)] = sq;
    data.parent_ddw_this[(1, 0)] = -sq;
    data.parent_ddw_this[(1, 1)] = -cq;

    data.this_w_parent[(0, 0)] = cq;
    data.this_w_parent[(0, 1)] = sq;
    data.this_w_parent[(1, 0)] = -sq;
    data.this_w_parent[(1, 1)] = cq;

    data.this_dw_parent[(0, 0)] = -sq;
    data.this_dw_parent[(0, 1)] = cq;
    data.this_dw_parent[(1, 0)] = -cq;
    data.this_dw_parent[(1, 1)] = -sq;

    data.this_ddw_parent[(0, 0)] = -cq;
    data.this_ddw_parent[(0, 1)] = -sq;
    data.this_ddw_parent[(1, 0)] = sq;
    data.this_ddw_parent[(1, 1)] = -cq;

    data.parent_z_this = data.parent_w_this.transpose() * data.parent_dw_this;
}

pub struct Frame {
    parent: Option<Weak<RefCell<Frame>>>,
    children: Vec<FrameRef>,
    coordinate_value: [f64; 2],
    coordinate_type: CoordinateType,
    is_fixed: bool,

    // Transformation matrix
    local_w: Matrix4d,
    global_w: Matrix4d,

    // First partial with respect this frame
    local_dw: Matrix4d,

    // Second partial with respect to this frame only
    local_ddw: Matrix4d,

    // Inverses of various matrices
    local_w_inv: Matrix4d,
    local_dw_inv: Matrix4d,
    local_ddw_inv: Matrix4d,

    // Twist matrix
    local_z: Matrix4d,

    // Frame velocity
    global_v: Matrix4d,
}

impl Frame {
    pub fn new(
        parent_frame: Option<&FrameRef>,
        coordinate_value: [f64; 2],
        coordinate_type: CoordinateType,
        is_coordinate_fixed: bool,
    ) -> FrameRef {
        let frame = Rc::new(RefCell::new(Self {
            parent: parent_frame.map(Rc::downgrade),
            children: Vec::new(),
            coordinate_value,
            coordinate_type,
            is_fixed: is_coordinate_fixed,
            local_w: Matrix4d::zeros(),
            global_w: Matrix4d::zeros(),
            local_dw: Matrix4d::zeros(),
            local_ddw: Matrix4d::zeros(),
            local_w_inv: Matrix4d::zeros(),
            local_dw_inv: Matrix4d::zeros(),
            local_ddw_inv: Matrix4d::zeros(),
            local_z: Matrix4d::zeros(),
            global_v: Matrix4d::zeros(),
        }));

        Self::update(&frame);

        if let Some(parent_frame) = parent_frame {
            parent_frame.borrow_mut().children.push(Rc::clone(&frame));
        }

        frame
    }

    fn parent_frame(this: &FrameRef) -> Option<FrameRef> {
        this.borrow().parent.as_ref().and_then(Weak::upgrade)
    }

    /// Update all cached matrices and their children. For best effect,
    /// only call this function from the "spatial frame" (the root frame).
    pub fn update(this: &FrameRef) {
        Self::update_local(this);

        if let Some(parent_frame) = Self::parent_frame(this) {
            let parent_global_w = parent_frame.borrow().global_w;
            let parent_global_v = parent_frame.borrow().global_v;

            let (local_w, local_w_inv, local_z, q_dot) = {
                let frame = this.borrow();
                (
                    frame.local_w,
                    frame.local_w_inv,
                    frame.local_z,
                    frame.coordinate_value[1],
                )
            };

            let global_w = parent_global_w * local_w;
            let global_v = local_w_inv * parent_global_v * local_w + local_z * q_dot;

            let mut frame = this.borrow_mut();
            frame.global_w = global_w;
            frame.global_v = global_v;
        } else {
            let local_w = this.borrow().local_w;
            let mut frame = this.borrow_mut();
            frame.global_w = local_w;
            frame.global_v = Matrix4d::zeros();
        }

        let children = this.borrow().children.clone();
        for child in children {
            Self::update(&child);
        }
    }

    /// Update the cached local transformation matrices.
    pub fn update_local(this: &FrameRef) {
        let (q, coordinate_type, is_fixed) = {
            let frame = this.borrow();
            (
                frame.coordinate_value[0],
                frame.coordinate_type,
                frame.is_fixed,
            )
        };
        if !is_fixed {
            let mut data = FrameData::default();
            match coordinate_type {
                CoordinateType::None => (),
                CoordinateType::XTran => update_translation_x(&mut data, q),
                CoordinateType::YTran => update_translation_y(&mut data, q),
                CoordinateType::ZTran => update_translation_z(&mut data, q),
                CoordinateType::XRot => update_rotation_x(&mut data, q),
                CoordinateType::YRot => update_rotation_y(&mut data, q),
                CoordinateType::ZRot => update_rotation_z(&mut data, q),
            }
            let mut frame = this.borrow_mut();
            frame.local_w = data.parent_w_this;
            frame.local_dw = data.parent_dw_this;
            frame.local_ddw = data.parent_ddw_this;
            frame.local_w_inv = data.this_w_parent;

            frame.local_dw_inv = data.this_dw_parent;
            frame.local_ddw_inv = data.this_ddw_parent;
            frame.local_z = data.parent_z_this;
        }
    }

    /// Transformation: first partial with respect to q in global coordinates.
    pub fn partial_w(this: &FrameRef, i_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            if Rc::ptr_eq(this, i_frame) {
                parent_frame.borrow().global_w * this.borrow().local_dw
            } else {
                Self::partial_w(&parent_frame, i_frame) * this.borrow().local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Transformation: second partial with respect to q in global coordinates.
    pub fn partial2_w(this: &FrameRef, i_frame: &FrameRef, j_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            let is_i = Rc::ptr_eq(this, i_frame);
            let is_j = Rc::ptr_eq(this, j_frame);

            if is_i && is_j {
                parent_frame.borrow().global_w * this.borrow().local_ddw
            } else if is_i {
                Self::partial_w(&parent_frame, j_frame) * this.borrow().local_dw
            } else if is_j {
                Self::partial_w(&parent_frame, i_frame) * this.borrow().local_dw
            } else {
                Self::partial2_w(&parent_frame, i_frame, j_frame) * this.borrow().local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Rigid body velocity: first partial with respect to q.
    pub fn partial_v(this: &FrameRef, i_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            let frame = this.borrow();

            if Rc::ptr_eq(this, i_frame) {
                frame.local_dw_inv * parent_frame.borrow().global_v * frame.local_w
                    + frame.local_w_inv * parent_frame.borrow().global_v * frame.local_dw
            } else {
                frame.local_w_inv * Self::partial_v(&parent_frame, i_frame) * frame.local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Rigid body velocity: second partial with respect to q.
    pub fn partial2_v(this: &FrameRef, i_frame: &FrameRef, j_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            let is_i = Rc::ptr_eq(this, i_frame);
            let is_j = Rc::ptr_eq(this, j_frame);
            let frame = this.borrow();

            if is_i && is_j {
                frame.local_ddw_inv * parent_frame.borrow().global_v * frame.local_w
                    + (frame.local_dw_inv * parent_frame.borrow().global_v * frame.local_dw) * 2.0
                    + frame.local_w_inv * parent_frame.borrow().global_v * frame.local_ddw
            } else if is_i {
                frame.local_dw_inv * Self::partial_v(&parent_frame, j_frame) * frame.local_w
                    + frame.local_w_inv * Self::partial_v(&parent_frame, j_frame) * frame.local_dw
            } else if is_j {
                frame.local_dw_inv * Self::partial_v(&parent_frame, i_frame) * frame.local_w
                    + frame.local_w_inv * Self::partial_v(&parent_frame, i_frame) * frame.local_dw
            } else {
                frame.local_w_inv
                    * Self::partial2_v(&parent_frame, i_frame, j_frame)
                    * frame.local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Rigid body velocity: first partial with respect to q_dot.
    pub fn partial_vd(this: &FrameRef, i_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            let frame = this.borrow();

            if Rc::ptr_eq(this, i_frame) {
                frame.local_z
            } else {
                frame.local_w_inv * Self::partial_vd(&parent_frame, i_frame) * frame.local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Rigid body velocity: second partial with respect to q_dot.
    pub fn partial2_vd(_this: &FrameRef, _i_frame: &FrameRef, _j_frame: &FrameRef) -> Matrix4d {
        Matrix4d::zeros()
    }

    /// Rigid body velocity: mixed partial with respect to q and q_dot.
    pub fn partial_v_mixed(this: &FrameRef, qdot_frame: &FrameRef, q_frame: &FrameRef) -> Matrix4d {
        if let Some(parent_frame) = Self::parent_frame(this) {
            let frame = this.borrow();

            if Rc::ptr_eq(this, qdot_frame) {
                Matrix4d::zeros()
            } else if Rc::ptr_eq(this, q_frame) {
                frame.local_dw_inv * Self::partial_vd(&parent_frame, qdot_frame) * frame.local_w
                    + frame.local_w_inv
                        * Self::partial_vd(&parent_frame, qdot_frame)
                        * frame.local_dw
            } else {
                frame.local_w_inv
                    * Self::partial_v_mixed(&parent_frame, qdot_frame, q_frame)
                    * frame.local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    pub fn children(this: &FrameRef) -> Vec<FrameRef> {
        this.borrow().children.clone()
    }

    pub fn set_coordinate_value(this: &FrameRef, coordinate_value: [f64; 2]) {
        this.borrow_mut().coordinate_value = coordinate_value;
    }

    pub fn local_w(this: &FrameRef) -> Matrix4d {
        this.borrow().local_w
    }

    pub fn global_w(this: &FrameRef) -> Matrix4d {
        this.borrow().global_w
    }

    pub fn local_dw(this: &FrameRef) -> Matrix4d {
        this.borrow().local_dw
    }

    pub fn local_ddw(this: &FrameRef) -> Matrix4d {
        this.borrow().local_ddw
    }

    pub fn global_v(this: &FrameRef) -> Matrix4d {
        this.borrow().global_v
    }

    pub fn local_dw_inv(this: &FrameRef) -> Matrix4d {
        this.borrow().local_dw_inv
    }

    pub fn local_ddw_inv(this: &FrameRef) -> Matrix4d {
        this.borrow().local_ddw_inv
    }

    pub fn local_z(this: &FrameRef) -> Matrix4d {
        this.borrow().local_z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_close(a: &Matrix4d, b: &Matrix4d) {
        let tol = 1.0e-12;
        for i in 0..4 {
            for j in 0..4 {
                let diff = (a[(i, j)] - b[(i, j)]).abs();
                assert!(diff <= tol, "matrix mismatch at ({i}, {j}): {diff}");
            }
        }
    }

    #[test]
    fn new_links_parent_and_updates_global_transform() {
        let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, false);
        let body_frame = Frame::new(Some(&world_frame), [2.0, 0.0], CoordinateType::XTran, false);

        assert_eq!(Frame::children(&world_frame).len(), 1);

        let mut world_tfm_body = Matrix4d::identity();
        world_tfm_body[(0, 3)] = 2.0;

        assert_matrix_close(&Frame::global_w(&body_frame), &world_tfm_body);
    }

    #[test]
    fn fixed_coordinate_clears_local_derivative_caches() {
        let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, false);
        let body_frame = Frame::new(Some(&world_frame), [0.2, 0.0], CoordinateType::YRot, true);

        assert_matrix_close(&Frame::local_dw(&body_frame), &Matrix4d::zeros());
        assert_matrix_close(&Frame::local_ddw(&body_frame), &Matrix4d::zeros());
        assert_matrix_close(&Frame::local_dw_inv(&body_frame), &Matrix4d::zeros());
        assert_matrix_close(&Frame::local_ddw_inv(&body_frame), &Matrix4d::zeros());
    }

    #[test]
    fn partial_w_matches_local_dw_for_direct_child_of_root() {
        let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, false);
        let body_frame = Frame::new(Some(&world_frame), [1.5, 0.0], CoordinateType::ZTran, false);

        assert_matrix_close(
            &Frame::partial_w(&body_frame, &body_frame),
            &Frame::local_dw(&body_frame),
        );
    }

    #[test]
    fn partial_vd_matches_local_z_for_self_frame() {
        let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, false);
        let body_frame = Frame::new(Some(&world_frame), [0.3, 0.0], CoordinateType::XRot, false);

        assert_matrix_close(
            &Frame::partial_vd(&body_frame, &body_frame),
            &Frame::local_z(&body_frame),
        );
    }

    #[test]
    fn update_from_root_propagates_to_descendants() {
        let world_frame = Frame::new(None, [0.0, 0.0], CoordinateType::None, false);
        let body_frame = Frame::new(Some(&world_frame), [1.0, 0.0], CoordinateType::XTran, false);
        let sensor_frame = Frame::new(Some(&body_frame), [2.0, 0.0], CoordinateType::YTran, false);

        Frame::set_coordinate_value(&body_frame, [3.0, 0.0]);
        Frame::update(&world_frame);

        let mut world_tfm_sensor = Matrix4d::identity();
        world_tfm_sensor[(0, 3)] = 3.0;
        world_tfm_sensor[(1, 3)] = 2.0;

        assert_matrix_close(&Frame::global_w(&sensor_frame), &world_tfm_sensor);
    }
}
