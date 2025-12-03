// src/ops.rs

use crate::tensor::{Tensor, TensorId, Shape};


/// Estrategia para combinar dos tensores en una operación elemento a elemento.
/// Soporta:
/// - escalar con cualquier shape (broadcast)
/// - vectores (rank 1) de distinta longitud (padding con neutros)
fn elementwise_binary_op(
    a: &Tensor,
    b: &Tensor,
    new_id: TensorId,
    neutral_a: f32,
    neutral_b: f32,
    op: impl Fn(f32, f32) -> Result<f32, String>,
) -> Result<Tensor, String> {
    match (a.shape.rank(), b.shape.rank()) {
        // Escalar con cualquier shape (broadcast)
        (0, _) => {
            let shape = b.shape.clone();
            let len = b.len();
            let mut data = Vec::with_capacity(len);
            let scalar = a.data[0];
            for &y in b.data.iter() {
                data.push(op(scalar, y)?);
            }
            Tensor::new(new_id, shape, data)
        }
        (_, 0) => {
            let shape = a.shape.clone();
            let len = a.len();
            let mut data = Vec::with_capacity(len);
            let scalar = b.data[0];
            for &x in a.data.iter() {
                data.push(op(x, scalar)?);
            }
            Tensor::new(new_id, shape, data)
        }
        // Vectores (rank 1) – posiblemente longitudes distintas
        (1, 1) => {
            let len_a = a.len();
            let len_b = b.len();
            let len = len_a.max(len_b);

            let mut data = Vec::with_capacity(len);

            for i in 0..len {
                let x = if i < len_a { a.data[i] } else { neutral_a };
                let y = if i < len_b { b.data[i] } else { neutral_b };
                data.push(op(x, y)?);
            }

            let shape = Shape::new(vec![len]);
            Tensor::new(new_id, shape, data)
        }
        // Otros casos: de momento, error
        _ => Err(format!(
            "Unsupported shapes for element-wise op: {:?} vs {:?}",
            a.shape.dims, b.shape.dims
        )),
    }
}

/// Verifica que dos shapes sean iguales
fn ensure_same_shape(a: &Shape, b: &Shape) -> Result<(), String> {
    if a.dims != b.dims {
        Err(format!("Shape mismatch: {:?} vs {:?}", a.dims, b.dims))
    } else {
        Ok(())
    }
}

/// Suma elemento a elemento: a + b
pub fn add(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    ensure_same_shape(&a.shape, &b.shape)?;
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    Tensor::new(new_id, a.shape.clone(), data)
}

/// Resta elemento a elemento: a - b
pub fn sub(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    ensure_same_shape(&a.shape, &b.shape)?;
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x - y)
        .collect();
    Tensor::new(new_id, a.shape.clone(), data)
}

/// Multiplicación elemento a elemento: a * b
pub fn multiply(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    ensure_same_shape(&a.shape, &b.shape)?;
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    Tensor::new(new_id, a.shape.clone(), data)
}

/// División elemento a elemento: a / b
pub fn divide(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    ensure_same_shape(&a.shape, &b.shape)?;
    let mut data = Vec::with_capacity(a.data.len());
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        if *y == 0.0 {
            return Err("Division by zero in element-wise divide".into());
        }
        data.push(x / y);
    }
    Tensor::new(new_id, a.shape.clone(), data)
}

/// Multiplicación por escalar: s * a
pub fn scalar_mul(a: &Tensor, s: f32, new_id: TensorId) -> Result<Tensor, String> {
    let data: Vec<f32> = a.data.iter().map(|x| s * x).collect();
    Tensor::new(new_id, a.shape.clone(), data)
}

/// Producto punto entre dos tensores rank-1 (vectores)
pub fn dot_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    if a.shape.rank() != 1 || b.shape.rank() != 1 {
        return Err("dot_1d expects rank-1 tensors".into());
    }
    ensure_same_shape(&a.shape, &b.shape)?;

    let sum = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .sum();

    Ok(sum)
}

/// Norma L2 de un tensor rank-1
pub fn l2_norm_1d(a: &Tensor) -> Result<f32, String> {
    if a.shape.rank() != 1 {
        return Err("l2_norm_1d expects rank-1 tensor".into());
    }

    Ok(a.data.iter().map(|x| x * x).sum::<f32>().sqrt())
}

/// Distancia L2 entre dos tensores rank-1
pub fn distance_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    if a.shape.rank() != 1 || b.shape.rank() != 1 {
        return Err("distance_1d expects rank-1 tensors".into());
    }
    ensure_same_shape(&a.shape, &b.shape)?;

    let sum_sq: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();

    Ok(sum_sq.sqrt())
}

/// Similitud coseno entre dos tensores rank-1
pub fn cosine_similarity_1d(a: &Tensor, b: &Tensor) -> Result<f32, String> {
    let dot_ab = dot_1d(a, b)?;
    let norm_a = l2_norm_1d(a)?;
    let norm_b = l2_norm_1d(b)?;

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Cannot compute cosine similarity with zero-norm vector".into());
    }

    Ok(dot_ab / (norm_a * norm_b))
}

/// Normaliza un tensor rank-1 a norma 1 (L2)
pub fn normalize_1d(a: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    let norm = l2_norm_1d(a)?;
    if norm == 0.0 {
        return Err("Cannot normalize a zero vector".into());
    }
    let factor = 1.0 / norm;
    scalar_mul(a, factor, new_id)
}

/// Suma elemento a elemento (RELAXED):
/// - escalar con cualquier shape
/// - vectores de distinta longitud → padding con 0.0
pub fn add_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        0.0, // neutral_a
        0.0, // neutral_b
        |x, y| Ok(x + y),
    )
}

/// Resta elemento a elemento (RELAXED)
pub fn sub_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        0.0, // si falta a → 0 - y
        0.0, // si falta b → x - 0
        |x, y| Ok(x - y),
    )
}

/// Multiplicación elemento a elemento (RELAXED)
pub fn multiply_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        1.0, // si falta a → 1 * y = y
        1.0, // si falta b → x * 1 = x
        |x, y| Ok(x * y),
    )
}

/// División elemento a elemento (RELAXED)
pub fn divide_relaxed(a: &Tensor, b: &Tensor, new_id: TensorId) -> Result<Tensor, String> {
    elementwise_binary_op(
        a,
        b,
        new_id,
        1.0, // si falta a → 1 / y
        1.0, // si falta b → x / 1 = x
        |x, y| {
            if y == 0.0 {
                Err("Division by zero in element-wise divide".into())
            } else {
                Ok(x / y)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, TensorId, Shape};

    fn tensor_1d(id: u64, vals: Vec<f32>) -> Tensor {
        let shape = Shape::new(vec![vals.len()]);
        Tensor::new(TensorId(id), shape, vals).unwrap()
    }

    #[test]
    fn test_add_simple() {
        let a = tensor_1d(1, vec![1.0, 2.0, 3.0]);
        let b = tensor_1d(2, vec![4.0, 5.0, 6.0]);
        let result = add(&a, &b, TensorId(3)).unwrap();

        assert_eq!(result.shape.dims, vec![3]);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_multiply_and_divide() {
        let a = tensor_1d(1, vec![1.0, 2.0, 3.0]);
        let b = tensor_1d(2, vec![4.0, 5.0, 6.0]);

        let prod = multiply(&a, &b, TensorId(3)).unwrap();
        assert_eq!(prod.data, vec![4.0, 10.0, 18.0]);

        let ratio = divide(&b, &a, TensorId(4)).unwrap();
        assert_eq!(ratio.data, vec![4.0, 2.5, 2.0]);
    }

    #[test]
    fn test_cosine_and_distance_and_normalize() {
        let a = tensor_1d(1, vec![1.0, 0.0, 0.0]);
        let b = tensor_1d(2, vec![1.0, 1.0, 0.0]);

        let sim = cosine_similarity_1d(&a, &b).unwrap();
        let expected_sim = 1.0 / 2f32.sqrt();
        assert!((sim - expected_sim).abs() < 1e-6);

        let dist = distance_1d(&a, &b).unwrap();
        let expected_dist = ((1.0_f32 - 1.0_f32).powi(2) + (0.0_f32 - 1.0_f32).powi(2) + 0.0f32).sqrt();
        assert!((dist - expected_dist).abs() < 1e-6);

        let n = normalize_1d(&b, TensorId(3)).unwrap();
        let norm_n = l2_norm_1d(&n).unwrap();
        assert!((norm_n - 1.0).abs() < 1e-6);
    }
}
