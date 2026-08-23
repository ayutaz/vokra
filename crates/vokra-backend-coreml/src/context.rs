//! Loaded CoreML model context and `MLMultiArray` feature binding.

use vokra_core::{Result, Tensor, VokraError};

use crate::artifact::{CoreMlArtifact, WHISPER_ENCODER_INPUT, WHISPER_ENCODER_OUTPUT};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod apple {
    use std::ffi::{CStr, CString};
    use std::marker::PhantomData;
    use std::rc::Rc;

    use super::*;
    use crate::sys::{self, Id};

    /// `MLMultiArrayDataTypeFloat32` (`MLMultiArray.h`).
    const ML_MULTI_ARRAY_F32: isize = 0x10000 | 32;
    /// `MLComputeUnitsCPUAndNeuralEngine` (`MLModelConfiguration.h`).
    const ML_COMPUTE_UNITS_CPU_AND_NEURAL_ENGINE: isize = 3;

    struct AutoreleasePool(*mut core::ffi::c_void);

    impl AutoreleasePool {
        unsafe fn push() -> Self {
            // SAFETY: paired with exactly one pop in Drop.
            Self(unsafe { sys::objc_autoreleasePoolPush() })
        }
    }

    impl Drop for AutoreleasePool {
        fn drop(&mut self) {
            // SAFETY: this token came from the matching pool push and is popped once.
            unsafe { sys::objc_autoreleasePoolPop(self.0) };
        }
    }

    /// A retained `MLModel` bound to one compiled Whisper encoder artifact.
    #[derive(Debug)]
    pub(crate) struct CoreMlContext {
        model: Id,
        artifact: CoreMlArtifact,
        // Objective-C/CoreML model objects are thread/queue-affine unless the
        // surrounding execution policy proves otherwise. Keep this context
        // explicitly !Send + !Sync, matching the Metal context posture.
        _not_send_sync: PhantomData<Rc<()>>,
    }

    impl CoreMlContext {
        pub(crate) fn load(artifact: CoreMlArtifact) -> Result<Self> {
            let path = artifact.compiled_model();
            if !path.is_dir() {
                return Err(VokraError::ModelLoad(format!(
                    "CoreML compiled artifact `{}` is missing or is not a .mlmodelc directory",
                    path.display()
                )));
            }

            // SAFETY: every selector and its exact typed objc_msgSend signature
            // is named at the call site. All autoreleased temporaries stay
            // inside this pool; the returned model is retained before pop.
            unsafe {
                let _pool = AutoreleasePool::push();
                let config_class = required_class(b"MLModelConfiguration\0")?;
                let model_class = required_class(b"MLModel\0")?;
                let ns_string = required_class(b"NSString\0")?;
                let ns_url = required_class(b"NSURL\0")?;

                // `+[MLModelConfiguration new] -> MLModelConfiguration *` (+1).
                let config = sys::send_id(config_class, sys::sel(b"new\0"));
                if config.is_null() {
                    return Err(VokraError::BackendUnavailable(
                        "CoreML failed to allocate MLModelConfiguration".to_owned(),
                    ));
                }
                // `-[MLModelConfiguration setComputeUnits:]`, NSInteger = 3.
                sys::send_void_isize(
                    config,
                    sys::sel(b"setComputeUnits:\0"),
                    ML_COMPUTE_UNITS_CPU_AND_NEURAL_ENGINE,
                );

                let path_utf8 = path.to_str().ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "CoreML artifact path is not valid UTF-8: {}",
                        path.display()
                    ))
                })?;
                let path_c = CString::new(path_utf8).map_err(|_| {
                    VokraError::InvalidArgument(format!(
                        "CoreML artifact path contains an interior NUL: {}",
                        path.display()
                    ))
                })?;
                // `+[NSString stringWithUTF8String:]` (autoreleased).
                let path_string = sys::send_id_cstr(
                    ns_string,
                    sys::sel(b"stringWithUTF8String:\0"),
                    path_c.as_ptr(),
                );
                if path_string.is_null() {
                    sys::send_void(config, sys::sel(b"release\0"));
                    return Err(VokraError::InvalidArgument(format!(
                        "CoreML could not represent artifact path `{}` as NSString",
                        path.display()
                    )));
                }
                // `+[NSURL fileURLWithPath:isDirectory:]` (autoreleased).
                let url = sys::send_id_id_bool(
                    ns_url,
                    sys::sel(b"fileURLWithPath:isDirectory:\0"),
                    path_string,
                    true,
                );
                if url.is_null() {
                    sys::send_void(config, sys::sel(b"release\0"));
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML could not create a file URL for `{}`",
                        path.display()
                    )));
                }

                let mut error: Id = core::ptr::null_mut();
                // `+[MLModel modelWithContentsOfURL:configuration:error:]`.
                // This synchronous API consumes an already-compiled
                // `.mlmodelc`; portable `.mlpackage` compilation remains an
                // offline converter step.
                let model = sys::send_id_id_id_error(
                    model_class,
                    sys::sel(b"modelWithContentsOfURL:configuration:error:\0"),
                    url,
                    config,
                    &mut error,
                );
                sys::send_void(config, sys::sel(b"release\0"));
                if model.is_null() {
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML failed to load `{}`: {}",
                        path.display(),
                        error_text(error)
                    )));
                }
                // Factory result is autoreleased; retain it across pool pop.
                let model = sys::send_id(model, sys::sel(b"retain\0"));
                Ok(Self {
                    model,
                    artifact,
                    _not_send_sync: PhantomData,
                })
            }
        }

        pub(crate) fn artifact(&self) -> &CoreMlArtifact {
            &self.artifact
        }

        pub(crate) fn predict_whisper_encoder(&self, input: &Tensor) -> Result<Tensor> {
            let want_input = self.artifact.input_shape();
            if input.shape.as_slice() != want_input {
                return Err(VokraError::InvalidArgument(format!(
                    "CoreML Whisper encoder input shape {:?} != artifact contract {want_input:?}",
                    input.shape
                )));
            }
            let input_data = input.as_f32()?;

            // SAFETY: selectors and exact send signatures are named at every
            // call. Owned alloc/init objects are released before pool pop.
            unsafe {
                let _pool = AutoreleasePool::push();
                let multi = make_multi_array(want_input, input_data)?;
                let feature_value_class = required_class(b"MLFeatureValue\0")?;
                // `+[MLFeatureValue featureValueWithMultiArray:]`.
                let feature_value = sys::send_id_id(
                    feature_value_class,
                    sys::sel(b"featureValueWithMultiArray:\0"),
                    multi,
                );
                if feature_value.is_null() {
                    sys::send_void(multi, sys::sel(b"release\0"));
                    return Err(VokraError::ModelLoad(
                        "CoreML could not wrap the input MLMultiArray".to_owned(),
                    ));
                }

                let key = ns_string(WHISPER_ENCODER_INPUT)?;
                let dictionary_class = required_class(b"NSDictionary\0")?;
                let values = [feature_value];
                let keys = [key];
                // `+[NSDictionary dictionaryWithObjects:forKeys:count:]`.
                let dictionary = sys::send_id_ptr_ptr_usize(
                    dictionary_class,
                    sys::sel(b"dictionaryWithObjects:forKeys:count:\0"),
                    values.as_ptr(),
                    keys.as_ptr(),
                    1,
                );
                if dictionary.is_null() {
                    sys::send_void(multi, sys::sel(b"release\0"));
                    return Err(VokraError::ModelLoad(
                        "CoreML could not create the input feature dictionary".to_owned(),
                    ));
                }

                let provider_class = required_class(b"MLDictionaryFeatureProvider\0")?;
                // `[[MLDictionaryFeatureProvider alloc] initWithDictionary:error:]`.
                let provider_alloc = sys::send_id(provider_class, sys::sel(b"alloc\0"));
                let mut error: Id = core::ptr::null_mut();
                let provider = sys::send_id_id_error(
                    provider_alloc,
                    sys::sel(b"initWithDictionary:error:\0"),
                    dictionary,
                    &mut error,
                );
                if provider.is_null() {
                    sys::send_void(multi, sys::sel(b"release\0"));
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML rejected the input feature dictionary: {}",
                        error_text(error)
                    )));
                }

                error = core::ptr::null_mut();
                // `-[MLModel predictionFromFeatures:error:]`.
                let output_provider = sys::send_id_id_error(
                    self.model,
                    sys::sel(b"predictionFromFeatures:error:\0"),
                    provider,
                    &mut error,
                );
                sys::send_void(provider, sys::sel(b"release\0"));
                sys::send_void(multi, sys::sel(b"release\0"));
                if output_provider.is_null() {
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML Whisper encoder prediction failed: {}",
                        error_text(error)
                    )));
                }

                let output_key = ns_string(WHISPER_ENCODER_OUTPUT)?;
                // `-[MLFeatureProvider featureValueForName:]`.
                let output_value = sys::send_id_id(
                    output_provider,
                    sys::sel(b"featureValueForName:\0"),
                    output_key,
                );
                if output_value.is_null() {
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML output is missing feature `{WHISPER_ENCODER_OUTPUT}`"
                    )));
                }
                // `-[MLFeatureValue multiArrayValue]`.
                let output_multi = sys::send_id(output_value, sys::sel(b"multiArrayValue\0"));
                if output_multi.is_null() {
                    return Err(VokraError::ModelLoad(format!(
                        "CoreML output feature `{WHISPER_ENCODER_OUTPUT}` is not an MLMultiArray"
                    )));
                }
                read_multi_array(output_multi, self.artifact.output_shape())
            }
        }
    }

    impl Drop for CoreMlContext {
        fn drop(&mut self) {
            // SAFETY: `model` was retained exactly once in `load` and is
            // released exactly once here.
            unsafe { sys::send_void(self.model, sys::sel(b"release\0")) };
        }
    }

    unsafe fn required_class(name: &[u8]) -> Result<Id> {
        // SAFETY: caller supplies a NUL-terminated class-name literal.
        let class = unsafe { sys::class(name) };
        if class.is_null() {
            let printable = CStr::from_bytes_with_nul(name)
                .ok()
                .and_then(|v| v.to_str().ok())
                .unwrap_or("<invalid class name>");
            return Err(VokraError::BackendUnavailable(format!(
                "Objective-C class `{printable}` is unavailable"
            )));
        }
        Ok(class)
    }

    unsafe fn ns_string(value: &str) -> Result<Id> {
        let c = CString::new(value).map_err(|_| {
            VokraError::InvalidArgument("CoreML feature name contains an interior NUL".to_owned())
        })?;
        // SAFETY: NUL-terminated CString and verified NSString class.
        let string = unsafe {
            sys::send_id_cstr(
                required_class(b"NSString\0")?,
                sys::sel(b"stringWithUTF8String:\0"),
                c.as_ptr(),
            )
        };
        if string.is_null() {
            return Err(VokraError::InvalidArgument(format!(
                "CoreML could not represent feature name `{value}` as NSString"
            )));
        }
        Ok(string)
    }

    unsafe fn ns_number(value: usize) -> Result<Id> {
        let value = u64::try_from(value).map_err(|_| {
            VokraError::InvalidArgument(format!(
                "CoreML shape axis {value} does not fit unsigned long long"
            ))
        })?;
        // SAFETY: NSNumber class and `numberWithUnsignedLongLong:` signature
        // are verified from Foundation `NSValue.h`.
        Ok(unsafe {
            sys::send_id_u64(
                required_class(b"NSNumber\0")?,
                sys::sel(b"numberWithUnsignedLongLong:\0"),
                value,
            )
        })
    }

    unsafe fn ns_array(values: &[Id]) -> Result<Id> {
        // SAFETY: values points at `values.len()` valid Objective-C objects.
        let array = unsafe {
            sys::send_id_ptr_usize(
                required_class(b"NSArray\0")?,
                sys::sel(b"arrayWithObjects:count:\0"),
                values.as_ptr(),
                values.len(),
            )
        };
        if array.is_null() {
            return Err(VokraError::ModelLoad(
                "Foundation failed to construct NSArray".to_owned(),
            ));
        }
        Ok(array)
    }

    unsafe fn make_multi_array(shape: [usize; 3], data: &[f32]) -> Result<Id> {
        let want = shape.into_iter().try_fold(1usize, |acc, axis| {
            acc.checked_mul(axis).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "CoreML input shape {shape:?} element count overflows usize"
                ))
            })
        })?;
        if data.len() != want {
            return Err(VokraError::InvalidArgument(format!(
                "CoreML input data length {} != shape {shape:?} element count {want}",
                data.len()
            )));
        }
        // SAFETY: all three dimensions fit u64 and the NSNumber class /
        // selector contract is validated inside `ns_number`.
        let shape_objects = unsafe {
            [
                ns_number(shape[0])?,
                ns_number(shape[1])?,
                ns_number(shape[2])?,
            ]
        };
        // SAFETY: `shape_objects` contains three live Objective-C objects.
        let shape_array = unsafe { ns_array(&shape_objects)? };
        // SAFETY: the caller holds an autorelease pool and CoreML is linked;
        // `required_class` checks a missing runtime class before returning it.
        let multi_class = unsafe { required_class(b"MLMultiArray\0")? };
        // `[[MLMultiArray alloc] initWithShape:dataType:error:]`.
        // SAFETY: `multi_class` is the live MLMultiArray class object and
        // `alloc` has the no-argument Objective-C object return signature.
        let alloc = unsafe { sys::send_id(multi_class, sys::sel(b"alloc\0")) };
        let mut error: Id = core::ptr::null_mut();
        // SAFETY: `alloc` is an allocated MLMultiArray receiver, `shape_array`
        // is live, the dtype is NSInteger, and `error` is a writable NSError*.
        let multi = unsafe {
            sys::send_id_id_isize_error(
                alloc,
                sys::sel(b"initWithShape:dataType:error:\0"),
                shape_array,
                ML_MULTI_ARRAY_F32,
                &mut error,
            )
        };
        if multi.is_null() {
            // SAFETY: CoreML either left `error` nil or returned a live NSError
            // inside the caller's autorelease pool.
            let message = unsafe { error_text(error) };
            return Err(VokraError::ModelLoad(format!(
                "CoreML could not allocate input MLMultiArray: {message}"
            )));
        }
        // SAFETY: `multi` is a live rank-3 Float32 MLMultiArray and `data`
        // exactly matches the already-validated element count.
        if let Err(err) = unsafe { copy_into_multi_array(multi, shape, data) } {
            // SAFETY: `multi` is an owned alloc/init object and is released
            // exactly once on this error path.
            unsafe { sys::send_void(multi, sys::sel(b"release\0")) };
            return Err(err);
        }
        Ok(multi)
    }

    unsafe fn array_usizes(array: Id, field: &str) -> Result<Vec<usize>> {
        if array.is_null() {
            return Err(VokraError::ModelLoad(format!(
                "CoreML MLMultiArray `{field}` returned nil"
            )));
        }
        // SAFETY: `array` is a non-null NSArray returned by CoreML; `count`
        // returns NSUInteger.
        let count = unsafe { sys::send_usize(array, sys::sel(b"count\0")) };
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: `index < count`; `objectAtIndex:` returns an object id.
            let number = unsafe { sys::send_id_usize(array, sys::sel(b"objectAtIndex:\0"), index) };
            if number.is_null() {
                return Err(VokraError::ModelLoad(format!(
                    "CoreML MLMultiArray `{field}` contains nil at index {index}"
                )));
            }
            // SAFETY: CoreML shape/stride arrays contain NSNumber objects and
            // this selector returns unsigned long long.
            let value = unsafe { sys::send_u64(number, sys::sel(b"unsignedLongLongValue\0")) };
            out.push(usize::try_from(value).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "CoreML MLMultiArray `{field}` value {value} does not fit usize"
                ))
            })?);
        }
        Ok(out)
    }

    unsafe fn layout(multi: Id, expected_shape: [usize; 3]) -> Result<[usize; 3]> {
        // SAFETY: `multi` is a live MLMultiArray and `dataType` returns the
        // NSInteger-backed MLMultiArrayDataType.
        let dtype = unsafe { sys::send_isize(multi, sys::sel(b"dataType\0")) };
        if dtype != ML_MULTI_ARRAY_F32 {
            return Err(VokraError::UnsupportedOp(format!(
                "CoreML delegate requires Float32 MLMultiArray I/O, got dataType {dtype}"
            )));
        }
        // SAFETY: `multi` is live; `shape` returns NSArray<NSNumber *> and
        // `array_usizes` validates nil entries and integer conversion.
        let shape = unsafe { array_usizes(sys::send_id(multi, sys::sel(b"shape\0")), "shape")? };
        if shape.as_slice() != expected_shape {
            return Err(VokraError::ModelLoad(format!(
                "CoreML MLMultiArray shape {shape:?} != artifact contract {expected_shape:?}"
            )));
        }
        // SAFETY: `multi` is live; `strides` returns NSArray<NSNumber *> and
        // `array_usizes` validates the returned objects.
        let strides =
            unsafe { array_usizes(sys::send_id(multi, sys::sel(b"strides\0")), "strides")? };
        let strides: [usize; 3] = strides.try_into().map_err(|v: Vec<usize>| {
            VokraError::ModelLoad(format!(
                "CoreML MLMultiArray strides must have rank 3, got {v:?}"
            ))
        })?;
        Ok(strides)
    }

    unsafe fn copy_into_multi_array(multi: Id, shape: [usize; 3], data: &[f32]) -> Result<()> {
        // SAFETY: the caller supplies a live MLMultiArray; `layout` validates
        // dtype, shape, and stride rank before pointer arithmetic.
        let strides = unsafe { layout(multi, shape)? };
        // `MLMultiArray.dataPointer` is still public but marked for future
        // deprecation. The replacement APIs are block-only; using them from
        // raw Rust FFI requires the separate audited block-literal bridge.
        // SAFETY: `multi` is a live Float32 MLMultiArray and `dataPointer`
        // returns the storage pointer whose layout was validated above.
        let ptr = unsafe { sys::send_ptr(multi, sys::sel(b"dataPointer\0")) }.cast::<f32>();
        if ptr.is_null() {
            return Err(VokraError::ModelLoad(
                "CoreML input MLMultiArray dataPointer returned nil".to_owned(),
            ));
        }
        let mut linear = 0usize;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let offset = i * strides[0] + j * strides[1] + k * strides[2];
                    // SAFETY: CoreML supplied the pointer and strides for this
                    // allocated array; `(i,j,k)` is inside its validated shape.
                    unsafe { ptr.add(offset).write(data[linear]) };
                    linear += 1;
                }
            }
        }
        Ok(())
    }

    unsafe fn read_multi_array(multi: Id, shape: [usize; 3]) -> Result<Tensor> {
        // SAFETY: the caller supplies a live MLMultiArray; `layout` validates
        // dtype, shape, and stride rank before pointer arithmetic.
        let strides = unsafe { layout(multi, shape)? };
        // SAFETY: `multi` is a live Float32 MLMultiArray and `dataPointer`
        // returns the storage pointer whose layout was validated above.
        let ptr = unsafe { sys::send_ptr(multi, sys::sel(b"dataPointer\0")) }.cast::<f32>();
        if ptr.is_null() {
            return Err(VokraError::ModelLoad(
                "CoreML output MLMultiArray dataPointer returned nil".to_owned(),
            ));
        }
        let count = shape[0]
            .checked_mul(shape[1])
            .and_then(|v| v.checked_mul(shape[2]))
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "CoreML output shape {shape:?} element count overflows usize"
                ))
            })?;
        let mut data = Vec::with_capacity(count);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let offset = i * strides[0] + j * strides[1] + k * strides[2];
                    // SAFETY: CoreML supplied the pointer and strides for this
                    // prediction result; `(i,j,k)` is inside validated shape.
                    data.push(unsafe { ptr.add(offset).read() });
                }
            }
        }
        Tensor::host_f32(shape.to_vec(), data)
    }

    unsafe fn error_text(error: Id) -> String {
        if error.is_null() {
            return "CoreML returned nil without NSError".to_owned();
        }
        // `-[NSError localizedDescription] -> NSString *`, then
        // `-[NSString UTF8String] -> const char *`.
        // SAFETY: `error` is a live NSError; `localizedDescription` returns an
        // Objective-C object within the active autorelease pool.
        let description = unsafe { sys::send_id(error, sys::sel(b"localizedDescription\0")) };
        if description.is_null() {
            return "CoreML NSError has no localizedDescription".to_owned();
        }
        // SAFETY: `description` is a live NSString and `UTF8String` returns a
        // NUL-terminated borrowed C string or null.
        let ptr = unsafe { sys::send_cstr(description, sys::sel(b"UTF8String\0")) };
        if ptr.is_null() {
            return "CoreML NSError description is not UTF-8".to_owned();
        }
        // SAFETY: NSString guarantees a NUL-terminated pointer valid for the
        // lifetime of the string, which is alive within the autorelease pool.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) use apple::CoreMlContext;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
pub(crate) struct CoreMlContext {
    artifact: CoreMlArtifact,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl CoreMlContext {
    pub(crate) fn load(artifact: CoreMlArtifact) -> Result<Self> {
        if !artifact.compiled_model().is_dir() {
            return Err(VokraError::ModelLoad(format!(
                "CoreML compiled artifact `{}` is missing or is not a .mlmodelc directory",
                artifact.compiled_model().display()
            )));
        }
        Err(VokraError::BackendUnavailable(
            "CoreML backend is not compiled for this target (only macOS / iOS)".to_owned(),
        ))
    }

    pub(crate) fn artifact(&self) -> &CoreMlArtifact {
        &self.artifact
    }

    pub(crate) fn predict_whisper_encoder(&self, _input: &Tensor) -> Result<Tensor> {
        Err(VokraError::BackendUnavailable(
            "CoreML backend is not compiled for this target (only macOS / iOS)".to_owned(),
        ))
    }
}
