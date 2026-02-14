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

    // Lazily recompute derivative/inverse caches when requested.
    local_differentials_valid: bool,
}

impl Frame {
    #[inline(always)]
    fn z_xrot() -> Matrix4d {
        let mut z = Matrix4d::zeros();
        z[(1, 2)] = -1.0;
        z[(2, 1)] = 1.0;
        z
    }

    #[inline(always)]
    fn z_yrot() -> Matrix4d {
        let mut z = Matrix4d::zeros();
        z[(0, 2)] = 1.0;
        z[(2, 0)] = -1.0;
        z
    }

    #[inline(always)]
    fn z_zrot() -> Matrix4d {
        let mut z = Matrix4d::zeros();
        z[(0, 1)] = -1.0;
        z[(1, 0)] = 1.0;
        z
    }

    #[inline(always)]
    fn mul_transform(a_tfm_b: &Matrix4d, b_tfm_c: &Matrix4d) -> Matrix4d {
        let mut a_tfm_c = Matrix4d::identity();

        for i in 0..3 {
            for j in 0..3 {
                a_tfm_c[(i, j)] = a_tfm_b[(i, 0)] * b_tfm_c[(0, j)]
                    + a_tfm_b[(i, 1)] * b_tfm_c[(1, j)]
                    + a_tfm_b[(i, 2)] * b_tfm_c[(2, j)];
            }
            a_tfm_c[(i, 3)] = a_tfm_b[(i, 0)] * b_tfm_c[(0, 3)]
                + a_tfm_b[(i, 1)] * b_tfm_c[(1, 3)]
                + a_tfm_b[(i, 2)] * b_tfm_c[(2, 3)]
                + a_tfm_b[(i, 3)];
        }

        a_tfm_c
    }

    #[inline(always)]
    fn similarity_twist(local_w: &Matrix4d, parent_global_v: &Matrix4d) -> Matrix4d {
        let mut local_w_inv_parent_global_v_local_w = Matrix4d::zeros();

        let mut omega_t = [0.0; 3];
        for i in 0..3 {
            omega_t[i] = parent_global_v[(i, 0)] * local_w[(0, 3)]
                + parent_global_v[(i, 1)] * local_w[(1, 3)]
                + parent_global_v[(i, 2)] * local_w[(2, 3)]
                + parent_global_v[(i, 3)];
        }

        let mut omega_r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                omega_r[i][j] = parent_global_v[(i, 0)] * local_w[(0, j)]
                    + parent_global_v[(i, 1)] * local_w[(1, j)]
                    + parent_global_v[(i, 2)] * local_w[(2, j)];
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                local_w_inv_parent_global_v_local_w[(i, j)] = local_w[(0, i)] * omega_r[0][j]
                    + local_w[(1, i)] * omega_r[1][j]
                    + local_w[(2, i)] * omega_r[2][j];
            }
            local_w_inv_parent_global_v_local_w[(i, 3)] = local_w[(0, i)] * omega_t[0]
                + local_w[(1, i)] * omega_t[1]
                + local_w[(2, i)] * omega_t[2];
        }

        local_w_inv_parent_global_v_local_w
    }

    #[inline(always)]
    fn update_global(
        local_w: &Matrix4d,
        parent_global_w: &Matrix4d,
        parent_global_v: &Matrix4d,
        q_dot: f64,
        coordinate_type: CoordinateType,
    ) -> (Matrix4d, Matrix4d) {
        let global_w = Self::mul_transform(parent_global_w, local_w);
        let mut global_v = Self::similarity_twist(local_w, parent_global_v);
        match coordinate_type {
            CoordinateType::None => {}
            CoordinateType::XTran => global_v[(0, 3)] += q_dot,
            CoordinateType::YTran => global_v[(1, 3)] += q_dot,
            CoordinateType::ZTran => global_v[(2, 3)] += q_dot,
            CoordinateType::XRot => {
                global_v[(1, 2)] -= q_dot;
                global_v[(2, 1)] += q_dot;
            }
            CoordinateType::YRot => {
                global_v[(0, 2)] += q_dot;
                global_v[(2, 0)] -= q_dot;
            }
            CoordinateType::ZRot => {
                global_v[(0, 1)] -= q_dot;
                global_v[(1, 0)] += q_dot;
            }
        }
        (global_w, global_v)
    }

    #[inline(always)]
    fn update_local_fast_in_place(frame: &mut Frame) {
        let q = frame.coordinate_value[0];
        match frame.coordinate_type {
            CoordinateType::None => {
                frame.local_w = Matrix4d::identity();
            }
            CoordinateType::XTran => {
                let mut local_w = Matrix4d::identity();
                local_w[(0, 3)] = q;
                frame.local_w = local_w;
            }
            CoordinateType::YTran => {
                let mut local_w = Matrix4d::identity();
                local_w[(1, 3)] = q;
                frame.local_w = local_w;
            }
            CoordinateType::ZTran => {
                let mut local_w = Matrix4d::identity();
                local_w[(2, 3)] = q;
                frame.local_w = local_w;
            }
            CoordinateType::XRot => {
                let (sq, cq) = q.sin_cos();
                let mut local_w = Matrix4d::identity();
                local_w[(1, 1)] = cq;
                local_w[(1, 2)] = -sq;
                local_w[(2, 1)] = sq;
                local_w[(2, 2)] = cq;
                frame.local_w = local_w;
            }
            CoordinateType::YRot => {
                let (sq, cq) = q.sin_cos();
                let mut local_w = Matrix4d::identity();
                local_w[(0, 0)] = cq;
                local_w[(0, 2)] = sq;
                local_w[(2, 0)] = -sq;
                local_w[(2, 2)] = cq;
                frame.local_w = local_w;
            }
            CoordinateType::ZRot => {
                let (sq, cq) = q.sin_cos();
                let mut local_w = Matrix4d::identity();
                local_w[(0, 0)] = cq;
                local_w[(0, 1)] = -sq;
                local_w[(1, 0)] = sq;
                local_w[(1, 1)] = cq;
                frame.local_w = local_w;
            }
        }
        frame.local_differentials_valid = false;
    }

    fn ensure_local_differentials(this: &FrameRef) {
        if this.borrow().local_differentials_valid {
            return;
        }
        let mut frame = this.borrow_mut();
        if frame.local_differentials_valid {
            return;
        }
        Self::update_local_in_place(&mut frame);
    }

    #[inline(always)]
    unsafe fn frame_ptr_from_ref(frame_ref: &FrameRef) -> *mut Frame {
        // SAFETY: `Rc::as_ptr` yields a stable pointer to the owned `RefCell<Frame>`.
        unsafe { (*Rc::as_ptr(frame_ref)).as_ptr() }
    }

    unsafe fn update_with_parent_raw(
        frame_ptr: *mut Frame,
        parent_global_w: *const Matrix4d,
        parent_global_v: *const Matrix4d,
    ) {
        // SAFETY: The caller guarantees `frame_ptr` is valid for exclusive update traversal.
        let frame = unsafe { &mut *frame_ptr };
        // SAFETY: Parent pointers come from live ancestor frames during traversal.
        let parent_global_w = unsafe { &*parent_global_w };
        // SAFETY: Parent pointers come from live ancestor frames during traversal.
        let parent_global_v = unsafe { &*parent_global_v };
        Self::update_local_fast_in_place(frame);
        let q_dot = frame.coordinate_value[1];
        (frame.global_w, frame.global_v) = Self::update_global(
            &frame.local_w,
            parent_global_w,
            parent_global_v,
            q_dot,
            frame.coordinate_type,
        );

        let global_w = &frame.global_w as *const Matrix4d;
        let global_v = &frame.global_v as *const Matrix4d;
        let children_ptr = frame.children.as_ptr();
        let child_count = frame.children.len();

        for i in 0..child_count {
            // SAFETY: child index is in-bounds and children vec is not mutated during traversal.
            let child_ref = unsafe { &*children_ptr.add(i) };
            // SAFETY: recursive traversal keeps each node update single-threaded and exclusive.
            let child_ptr = unsafe { Self::frame_ptr_from_ref(child_ref) };
            // SAFETY: same traversal guarantees as above.
            unsafe { Self::update_with_parent_raw(child_ptr, global_w, global_v) };
        }
    }

    unsafe fn update_root_raw(this: &FrameRef) {
        // SAFETY: root pointer is valid for the duration of update traversal.
        let root_ptr = unsafe { Self::frame_ptr_from_ref(this) };
        // SAFETY: traversal performs exclusive mutable updates.
        let root = unsafe { &mut *root_ptr };
        Self::update_local_fast_in_place(root);
        root.global_w = root.local_w;
        root.global_v = Matrix4d::zeros();

        let root_global_w = &root.global_w as *const Matrix4d;
        let root_global_v = &root.global_v as *const Matrix4d;
        let children_ptr = root.children.as_ptr();
        let child_count = root.children.len();

        for i in 0..child_count {
            // SAFETY: child index is in-bounds and children vec is not mutated during traversal.
            let child_ref = unsafe { &*children_ptr.add(i) };
            // SAFETY: child pointer originates from valid `Rc<RefCell<Frame>>`.
            let child_ptr = unsafe { Self::frame_ptr_from_ref(child_ref) };
            // SAFETY: same traversal guarantees as above.
            unsafe { Self::update_with_parent_raw(child_ptr, root_global_w, root_global_v) };
        }
    }

    fn update_with_parent(this: &FrameRef, parent_global_w: Matrix4d, parent_global_v: Matrix4d) {
        let (children, global_w, global_v) = {
            let mut frame = this.borrow_mut();
            Self::update_local_fast_in_place(&mut frame);
            let q_dot = frame.coordinate_value[1];
            (frame.global_w, frame.global_v) = Self::update_global(
                &frame.local_w,
                &parent_global_w,
                &parent_global_v,
                q_dot,
                frame.coordinate_type,
            );
            (frame.children.clone(), frame.global_w, frame.global_v)
        };

        for child in children {
            Self::update_with_parent(&child, global_w, global_v);
        }
    }

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
            local_differentials_valid: false,
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
        if Self::parent_frame(this).is_none() {
            // SAFETY: root traversal is single-threaded and exclusively mutates cached fields.
            unsafe { Self::update_root_raw(this) };
            return;
        }

        if let Some(parent_frame) = Self::parent_frame(this) {
            let parent = parent_frame.borrow();
            let parent_global_w = parent.global_w;
            let parent_global_v = parent.global_v;
            drop(parent);

            let (children, global_w, global_v) = {
                let mut frame = this.borrow_mut();
                Self::update_local_fast_in_place(&mut frame);
                let q_dot = frame.coordinate_value[1];
                (frame.global_w, frame.global_v) = Self::update_global(
                    &frame.local_w,
                    &parent_global_w,
                    &parent_global_v,
                    q_dot,
                    frame.coordinate_type,
                );
                (frame.children.clone(), frame.global_w, frame.global_v)
            };

            for child in children {
                Self::update_with_parent(&child, global_w, global_v);
            }
        } else {
            let (children, local_w, global_v_zero) = {
                let mut frame = this.borrow_mut();
                Self::update_local_fast_in_place(&mut frame);
                frame.global_w = frame.local_w;
                frame.global_v = Matrix4d::zeros();
                (frame.children.clone(), frame.local_w, frame.global_v)
            };

            for child in children {
                Self::update_with_parent(&child, local_w, global_v_zero);
            }
        }
    }

    /// Update the cached local transformation matrices.
    pub fn update_local(this: &FrameRef) {
        let mut frame = this.borrow_mut();
        Self::update_local_in_place(&mut frame);
    }

    fn update_local_in_place(frame: &mut Frame) {
        let q = frame.coordinate_value[0];
        let (w, dw, ddw, w_inv, dw_inv, ddw_inv, z) = match frame.coordinate_type {
            // no transformation
            CoordinateType::None => (
                Matrix4d::identity(),
                Matrix4d::zeros(),
                Matrix4d::zeros(),
                Matrix4d::identity(),
                Matrix4d::zeros(),
                Matrix4d::zeros(),
                Matrix4d::zeros(),
            ),

            // x-translation
            CoordinateType::XTran => {
                let mut w = Matrix4d::identity();
                w[(0, 3)] = q;

                let mut dw = Matrix4d::zeros();
                dw[(0, 3)] = 1.0;

                let ddw = Matrix4d::zeros();

                let mut w_inv = w;
                w_inv[(0, 3)] = -q;

                let mut dw_inv = dw;
                dw_inv[(0, 3)] = -1.0;

                let ddw_inv = ddw;
                let z = dw;
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }

            // y-translation
            CoordinateType::YTran => {
                let mut w = Matrix4d::identity();
                w[(1, 3)] = q;

                let mut dw = Matrix4d::zeros();
                dw[(1, 3)] = 1.0;

                let ddw = Matrix4d::zeros();

                let mut w_inv = w;
                w_inv[(1, 3)] = -q;

                let mut dw_inv = dw;
                dw_inv[(1, 3)] = -1.0;

                let ddw_inv = ddw;
                let z = dw;
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }

            // z-translation
            CoordinateType::ZTran => {
                let mut w = Matrix4d::identity();
                w[(2, 3)] = q;

                let mut dw = Matrix4d::zeros();
                dw[(2, 3)] = 1.0;

                let ddw = Matrix4d::zeros();

                let mut w_inv = w;
                w_inv[(2, 3)] = -q;

                let mut dw_inv = dw;
                dw_inv[(2, 3)] = -1.0;

                let ddw_inv = ddw;
                let z = dw;
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }

            // x-rotation
            CoordinateType::XRot => {
                let (sq, cq) = q.sin_cos();

                let mut w = Matrix4d::identity();
                w[(1, 1)] = cq;
                w[(1, 2)] = -sq;
                w[(2, 1)] = sq;
                w[(2, 2)] = cq;

                let mut dw = Matrix4d::zeros();
                dw[(1, 1)] = -sq;
                dw[(1, 2)] = -cq;
                dw[(2, 1)] = cq;
                dw[(2, 2)] = -sq;

                let mut ddw = Matrix4d::zeros();
                ddw[(1, 1)] = -cq;
                ddw[(1, 2)] = sq;
                ddw[(2, 1)] = -sq;
                ddw[(2, 2)] = -cq;

                let mut w_inv = Matrix4d::identity();
                w_inv[(1, 1)] = cq;
                w_inv[(1, 2)] = sq;
                w_inv[(2, 1)] = -sq;
                w_inv[(2, 2)] = cq;

                let mut dw_inv = Matrix4d::zeros();
                dw_inv[(1, 1)] = -sq;
                dw_inv[(1, 2)] = cq;
                dw_inv[(2, 1)] = -cq;
                dw_inv[(2, 2)] = -sq;

                let mut ddw_inv = Matrix4d::zeros();
                ddw_inv[(1, 1)] = -cq;
                ddw_inv[(1, 2)] = -sq;
                ddw_inv[(2, 1)] = sq;
                ddw_inv[(2, 2)] = -cq;

                let z = Self::z_xrot();
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }

            // y-rotation
            CoordinateType::YRot => {
                let (sq, cq) = q.sin_cos();

                let mut w = Matrix4d::identity();
                w[(0, 0)] = cq;
                w[(0, 2)] = sq;
                w[(2, 0)] = -sq;
                w[(2, 2)] = cq;

                let mut dw = Matrix4d::zeros();
                dw[(0, 0)] = -sq;
                dw[(0, 2)] = cq;
                dw[(2, 0)] = -cq;
                dw[(2, 2)] = -sq;

                let mut ddw = Matrix4d::zeros();
                ddw[(0, 0)] = -cq;
                ddw[(0, 2)] = -sq;
                ddw[(2, 0)] = sq;
                ddw[(2, 2)] = -cq;

                let mut w_inv = Matrix4d::identity();
                w_inv[(0, 0)] = cq;
                w_inv[(0, 2)] = -sq;
                w_inv[(2, 0)] = sq;
                w_inv[(2, 2)] = cq;

                let mut dw_inv = Matrix4d::zeros();
                dw_inv[(0, 0)] = -sq;
                dw_inv[(0, 2)] = -cq;
                dw_inv[(2, 0)] = cq;
                dw_inv[(2, 2)] = -sq;

                let mut ddw_inv = Matrix4d::zeros();
                ddw_inv[(0, 0)] = -cq;
                ddw_inv[(0, 2)] = sq;
                ddw_inv[(2, 0)] = -sq;
                ddw_inv[(2, 2)] = -cq;

                let z = Self::z_yrot();
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }

            // z-rotation
            CoordinateType::ZRot => {
                let (sq, cq) = q.sin_cos();

                let mut w = Matrix4d::identity();
                w[(0, 0)] = cq;
                w[(0, 1)] = -sq;
                w[(1, 0)] = sq;
                w[(1, 1)] = cq;

                let mut dw = Matrix4d::zeros();
                dw[(0, 0)] = -sq;
                dw[(0, 1)] = -cq;
                dw[(1, 0)] = cq;
                dw[(1, 1)] = -sq;

                let mut ddw = Matrix4d::zeros();
                ddw[(0, 0)] = -cq;
                ddw[(0, 1)] = sq;
                ddw[(1, 0)] = -sq;
                ddw[(1, 1)] = -cq;

                let mut w_inv = Matrix4d::identity();
                w_inv[(0, 0)] = cq;
                w_inv[(0, 1)] = sq;
                w_inv[(1, 0)] = -sq;
                w_inv[(1, 1)] = cq;

                let mut dw_inv = Matrix4d::zeros();
                dw_inv[(0, 0)] = -sq;
                dw_inv[(0, 1)] = cq;
                dw_inv[(1, 0)] = -cq;
                dw_inv[(1, 1)] = -sq;

                let mut ddw_inv = Matrix4d::zeros();
                ddw_inv[(0, 0)] = -cq;
                ddw_inv[(0, 1)] = -sq;
                ddw_inv[(1, 0)] = sq;
                ddw_inv[(1, 1)] = -cq;

                let z = Self::z_zrot();
                (w, dw, ddw, w_inv, dw_inv, ddw_inv, z)
            }
        };

        frame.local_w = w;
        frame.local_w_inv = w_inv;
        frame.local_z = z;

        if frame.is_fixed {
            frame.local_dw = Matrix4d::zeros();
            frame.local_ddw = Matrix4d::zeros();
            frame.local_dw_inv = Matrix4d::zeros();
            frame.local_ddw_inv = Matrix4d::zeros();
        } else {
            frame.local_dw = dw;
            frame.local_ddw = ddw;
            frame.local_dw_inv = dw_inv;
            frame.local_ddw_inv = ddw_inv;
        }
        frame.local_differentials_valid = true;
    }

    /// Transformation: first partial with respect to q in global coordinates.
    pub fn partial_w(this: &FrameRef, i_frame: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
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
        Self::ensure_local_differentials(this);
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
        Self::ensure_local_differentials(this);
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
        Self::ensure_local_differentials(this);
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
                frame.local_w_inv * Self::partial2_v(&parent_frame, i_frame, j_frame) * frame.local_w
            }
        } else {
            Matrix4d::zeros()
        }
    }

    /// Rigid body velocity: first partial with respect to q_dot.
    pub fn partial_vd(this: &FrameRef, i_frame: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
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
        Self::ensure_local_differentials(this);
        if let Some(parent_frame) = Self::parent_frame(this) {
            let frame = this.borrow();

            if Rc::ptr_eq(this, qdot_frame) {
                Matrix4d::zeros()
            } else if Rc::ptr_eq(this, q_frame) {
                frame.local_dw_inv * Self::partial_vd(&parent_frame, qdot_frame) * frame.local_w
                    + frame.local_w_inv * Self::partial_vd(&parent_frame, qdot_frame) * frame.local_dw
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
        Self::ensure_local_differentials(this);
        this.borrow().local_dw
    }

    pub fn local_ddw(this: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
        this.borrow().local_ddw
    }

    pub fn global_v(this: &FrameRef) -> Matrix4d {
        this.borrow().global_v
    }

    pub fn local_dw_inv(this: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
        this.borrow().local_dw_inv
    }

    pub fn local_ddw_inv(this: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
        this.borrow().local_ddw_inv
    }

    pub fn local_z(this: &FrameRef) -> Matrix4d {
        Self::ensure_local_differentials(this);
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
