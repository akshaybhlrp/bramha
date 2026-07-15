use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Operator {
    #[serde(rename = "$eq")]
    Eq,
    #[serde(rename = "$ne")]
    Ne,
    #[serde(rename = "$gt")]
    Gt,
    #[serde(rename = "$lt")]
    Lt,
    #[serde(rename = "$gte")]
    Gte,
    #[serde(rename = "$lte")]
    Lte,
    #[serde(rename = "$in")]
    In,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComparisonFilter {
    pub field: String,
    pub operator: Operator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LogicalFilter {
    #[serde(rename = "$and")]
    And(Vec<Filter>),
    #[serde(rename = "$or")]
    Or(Vec<Filter>),
    #[serde(rename = "$not")]
    Not(Box<Filter>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Filter {
    Logical(LogicalFilter),
    Comparison(ComparisonFilter),
}

impl Filter {
    pub fn matches(&self, metadata: &Option<serde_json::Value>) -> bool {
        let meta_obj = match metadata {
            Some(serde_json::Value::Object(map)) => map,
            _ => return false, // If there's no metadata object, it matches nothing
        };

        match self {
            Filter::Logical(logical) => match logical {
                LogicalFilter::And(filters) => {
                    if filters.is_empty() {
                        return true;
                    }
                    filters.iter().all(|f| f.matches(metadata))
                }
                LogicalFilter::Or(filters) => {
                    if filters.is_empty() {
                        return false;
                    }
                    filters.iter().any(|f| f.matches(metadata))
                }
                LogicalFilter::Not(filter) => !filter.matches(metadata),
            },
            Filter::Comparison(comp) => {
                let field_value = match get_nested_value(meta_obj, &comp.field) {
                    Some(val) => val,
                    None => return comp.operator == Operator::Ne, // Field not existing means it matches $ne, but fails other ops
                };

                match comp.operator {
                    Operator::Eq => field_value == &comp.value,
                    Operator::Ne => field_value != &comp.value,
                    Operator::Gt => {
                        compare_values(field_value, &comp.value)
                            == Some(std::cmp::Ordering::Greater)
                    }
                    Operator::Lt => {
                        compare_values(field_value, &comp.value) == Some(std::cmp::Ordering::Less)
                    }
                    Operator::Gte => matches!(
                        compare_values(field_value, &comp.value),
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
                    ),
                    Operator::Lte => matches!(
                        compare_values(field_value, &comp.value),
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
                    ),
                    Operator::In => {
                        if let serde_json::Value::Array(arr) = &comp.value {
                            arr.contains(field_value)
                        } else {
                            false
                        }
                    }
                }
            }
        }
    }

    /// Compiles the Filter AST into a parameterized SQLite query utilizing JSON1 extraction.
    pub fn to_sql_query(&self) -> (String, Vec<serde_json::Value>) {
        match self {
            Filter::Logical(logical) => match logical {
                LogicalFilter::And(filters) => {
                    if filters.is_empty() {
                        return ("1=1".to_string(), vec![]);
                    }
                    let mut parts = Vec::new();
                    let mut params = Vec::new();
                    for f in filters {
                        let (q, p) = f.to_sql_query();
                        parts.push(format!("({})", q));
                        params.extend(p);
                    }
                    (parts.join(" AND "), params)
                }
                LogicalFilter::Or(filters) => {
                    if filters.is_empty() {
                        return ("1=0".to_string(), vec![]);
                    }
                    let mut parts = Vec::new();
                    let mut params = Vec::new();
                    for f in filters {
                        let (q, p) = f.to_sql_query();
                        parts.push(format!("({})", q));
                        params.extend(p);
                    }
                    (parts.join(" OR "), params)
                }
                LogicalFilter::Not(filter) => {
                    let (q, p) = filter.to_sql_query();
                    (format!("NOT ({})", q), p)
                }
            },
            Filter::Comparison(comp) => {
                let json_path = format!("$.{}", comp.field);
                let col = format!("json_extract(metadata, '{}')", json_path);

                match comp.operator {
                    Operator::Eq => (format!("{} = ?", col), vec![comp.value.clone()]),
                    Operator::Ne => (format!("{} != ?", col), vec![comp.value.clone()]),
                    Operator::Gt => (format!("{} > ?", col), vec![comp.value.clone()]),
                    Operator::Lt => (format!("{} < ?", col), vec![comp.value.clone()]),
                    Operator::Gte => (format!("{} >= ?", col), vec![comp.value.clone()]),
                    Operator::Lte => (format!("{} <= ?", col), vec![comp.value.clone()]),
                    Operator::In => {
                        if let serde_json::Value::Array(arr) = &comp.value {
                            let placeholders = vec!["?"; arr.len()].join(", ");
                            (format!("{} IN ({})", col, placeholders), arr.clone())
                        } else {
                            ("1=0".to_string(), vec![])
                        }
                    }
                }
            }
        }
    }
}

/// Helper to get nested values from JSON map (e.g. "author.name" or "year")
fn get_nested_value<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    field_path: &str,
) -> Option<&'a serde_json::Value> {
    let parts: Vec<&str> = field_path.split('.').collect();
    let mut current_val = Some(serde_json::Value::Object(map.clone()));

    for part in parts {
        current_val = match current_val {
            Some(serde_json::Value::Object(ref m)) => m.get(part).cloned(),
            _ => return None,
        };
    }

    // We need to return a reference to the actual value in the map, so we lookup in map directly if possible, or build it
    // To avoid lifetime issues since we cloned during descent, let's trace from raw reference:
    let mut current_ref = map.get(field_path.split('.').next()?);
    let rest_parts = &field_path.split('.').collect::<Vec<&str>>()[1..];

    for part in rest_parts {
        current_ref = match current_ref {
            Some(serde_json::Value::Object(m)) => m.get(*part),
            _ => return None,
        };
    }
    current_ref
}

/// Utility to compare two arbitrary JSON values numerically or lexicographically
fn compare_values(a: &serde_json::Value, b: &serde_json::Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            if let (Some(f1), Some(f2)) = (n1.as_f64(), n2.as_f64()) {
                f1.partial_cmp(&f2)
            } else {
                None
            }
        }
        (serde_json::Value::String(s1), serde_json::Value::String(s2)) => Some(s1.cmp(s2)),
        (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) => Some(b1.cmp(b2)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_comparison_filter() {
        let meta = Some(json!({
            "category": "finance",
            "year": 2024,
            "author": {
                "name": "Alice"
            }
        }));

        let filter_eq = Filter::Comparison(ComparisonFilter {
            field: "category".to_string(),
            operator: Operator::Eq,
            value: json!("finance"),
        });
        assert!(filter_eq.matches(&meta));

        let filter_gt = Filter::Comparison(ComparisonFilter {
            field: "year".to_string(),
            operator: Operator::Gt,
            value: json!(2020),
        });
        assert!(filter_gt.matches(&meta));

        let filter_nested = Filter::Comparison(ComparisonFilter {
            field: "author.name".to_string(),
            operator: Operator::Eq,
            value: json!("Alice"),
        });
        assert!(filter_nested.matches(&meta));
    }

    #[test]
    fn test_logical_and_filter() {
        let meta = Some(json!({
            "category": "finance",
            "year": 2024
        }));

        let filter_and = Filter::Logical(LogicalFilter::And(vec![
            Filter::Comparison(ComparisonFilter {
                field: "category".to_string(),
                operator: Operator::Eq,
                value: json!("finance"),
            }),
            Filter::Comparison(ComparisonFilter {
                field: "year".to_string(),
                operator: Operator::Lt,
                value: json!(2025),
            }),
        ]));

        assert!(filter_and.matches(&meta));
    }
}
