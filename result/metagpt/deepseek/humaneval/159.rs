#[derive(Debug, PartialEq)]
pub struct Result {
    pub total_eaten: i32,
    pub carrots_left: i32,
}

pub fn eat(number: i32, need: i32, remaining: i32) -> Result {
    if need > remaining {
        Result {
            total_eaten: number + remaining,
            carrots_left: 0,
        }
    } else {
        Result {
            total_eaten: number + need,
            carrots_left: remaining - need,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eat() {
        // 测试需求大于剩余的情况
        assert_eq!(
            eat(5, 10, 8),
            Result {
                total_eaten: 13,
                carrots_left: 0
            }
        );

        // 测试需求小于剩余的情况
        assert_eq!(
            eat(5, 3, 10),
            Result {
                total_eaten: 8,
                carrots_left: 7
            }
        );

        // 测试需求等于剩余的情况
        assert_eq!(
            eat(5, 5, 5),
            Result {
                total_eaten: 10,
                carrots_left: 0
            }
        );

        // 测试边界情况
        assert_eq!(
            eat(0, 0, 0),
            Result {
                total_eaten: 0,
                carrots_left: 0
            }
        );
    }
}