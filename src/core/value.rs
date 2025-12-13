// src/value.rs

use serde::Serialize;
use std::fmt;

/// Represents a value in the database - supports heterogeneous types
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Value {
    Float(f32),
    Int(i64),
    String(String),
    Bool(bool),
    Vector(Vec<f32>), // Embedding vector
    Null,
}

/// Type descriptor for values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum ValueType {
    Float,
    Int,
    String,
    Bool,
    Vector(usize), // Vector with fixed dimension
    Null,
}

impl Value {
    /// Get the type of this value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Float(_) => ValueType::Float,
            Value::Int(_) => ValueType::Int,
            Value::String(_) => ValueType::String,
            Value::Bool(_) => ValueType::Bool,
            Value::Vector(v) => ValueType::Vector(v.len()),
            Value::Null => ValueType::Null,
        }
    }

    /// Check if this value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to convert to f32
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Try to convert to i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to get string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get vector reference
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Compare values (for sorting and filtering)
    pub fn compare(&self, other: &Value) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) => Some(Ordering::Less),
            (_, Value::Null) => Some(Ordering::Greater),
            // Cross-type numeric comparison
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f32)),
            (Value::Int(a), Value::Float(b)) => (*a as f32).partial_cmp(b),
            _ => None,
        }
    }

    /// Check if this value matches the given type
    pub fn matches_type(&self, value_type: &ValueType) -> bool {
        match (self, value_type) {
            (Value::Float(_), ValueType::Float) => true,
            (Value::Int(_), ValueType::Int) => true,
            (Value::String(_), ValueType::String) => true,
            (Value::Bool(_), ValueType::Bool) => true,
            (Value::Vector(v), ValueType::Vector(dim)) => v.len() == *dim,
            (Value::Null, _) => true, // Null matches any type if nullable
            _ => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{}", v),
            Value::Int(v) => write!(f, "{}", v),
            Value::String(v) => write!(f, "\"{}\"", v),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Vector(v) => {
                write!(f, "[")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                    if i >= 5 && v.len() > 7 {
                        write!(f, ", ... ({} more)", v.len() - 6)?;
                        break;
                    }
                }
                write!(f, "]")
            }
            Value::Null => write!(f, "NULL"),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Float => write!(f, "FLOAT"),
            ValueType::Int => write!(f, "INT"),
            ValueType::String => write!(f, "STRING"),
            ValueType::Bool => write!(f, "BOOL"),
            ValueType::Vector(dim) => write!(f, "VECTOR[{}]", dim),
            ValueType::Null => write!(f, "NULL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_types() {
        assert_eq!(Value::Float(1.5).value_type(), ValueType::Float);
        assert_eq!(Value::Int(42).value_type(), ValueType::Int);
        assert_eq!(
            Value::String("hello".to_string()).value_type(),
            ValueType::String
        );
        assert_eq!(Value::Bool(true).value_type(), ValueType::Bool);
        assert_eq!(
            Value::Vector(vec![1.0, 2.0, 3.0]).value_type(),
            ValueType::Vector(3)
        );
        assert_eq!(Value::Null.value_type(), ValueType::Null);
    }

    #[test]
    fn test_value_conversions() {
        let float_val = Value::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_int(), Some(3));

        let int_val = Value::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let str_val = Value::String("test".to_string());
        assert_eq!(str_val.as_str(), Some("test"));
        assert_eq!(str_val.as_int(), None);
    }

    #[test]
    fn test_value_comparison() {
        use std::cmp::Ordering;

        assert_eq!(Value::Int(5).compare(&Value::Int(10)), Some(Ordering::Less));
        assert_eq!(
            Value::Float(3.14).compare(&Value::Float(2.71)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Value::String("a".to_string()).compare(&Value::String("b".to_string())),
            Some(Ordering::Less)
        );

        // Cross-type numeric comparison
        assert_eq!(
            Value::Int(5).compare(&Value::Float(5.0)),
            Some(Ordering::Equal)
        );

        // Null handling
        assert_eq!(Value::Null.compare(&Value::Int(5)), Some(Ordering::Less));
        assert_eq!(Value::Int(5).compare(&Value::Null), Some(Ordering::Greater));
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::Float(3.14).to_string(), "3.14");
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::String("hello".to_string()).to_string(), "\"hello\"");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Null.to_string(), "NULL");

        let vec_val = Value::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec_val.to_string(), "[1, 2, 3]");
    }
}
