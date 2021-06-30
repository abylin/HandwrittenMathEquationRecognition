import re

multi_div_regular = re.compile("\d+\.*\d*[\*\/]+[\-]?\d+\.*\d*")
add_minus_regular = re.compile("[\-]?\d+\.*\d*[\+\-]{1}\d+\.*\d*")
parentheses_regular = re.compile(r"\(([^()]+)\)")

# Do multiplication and division calculations
def multi_div_compute(str_expire):
    mat = multi_div_regular.search(str_expire) 
    if not mat: 
        return str_expire
    match_data = multi_div_regular.search(str_expire).group() 
    if len(match_data.split("*")) > 1:   # There's an * notation, so we can to the multiplication。
        left_part, right_part2 = match_data.split("*")   
        value = float(left_part) * float(right_part2)   
    else:
        left_part, right_part2 = match_data.split("/")   # The program does division.
        if float(right_part2) == 0:   
            raise Exception("The dividend cannot be 0.")
        value = float(left_part) / float(right_part2)   
    # Replace the result of the evaluation back into the expression to generate a new expression
    s1, s2 = multi_div_regular.split(str_expire, 1)  
    new_expression = "%s%s%s" % (s1, value, s2)
    return multi_div_compute(new_expression) 

# Do addition and subtraction calculations
def add_minus_compute(str_expire):
    str_expire = str_expire.replace("+-", "-") # example 9+-3 => 9-3
    str_expire = str_expire.replace("--", "+") # example 9--3 => 9+3
    mat = add_minus_regular.search(str_expire) 
    # If there is no plus or minus sign, the expression is proved to have been evaluated and the final result is returned.
    if not mat:   
        return str_expire
    # If the equation starts with a minus sign, it means the number on the left is negative
    left_sign = 1
    if str_expire.startswith('-'):   
        left_sign = -1
        str_expire = str_expire[1:]
    match_data = add_minus_regular.search(str_expire).group()  
    if len(match_data.split('+')) > 1:   # There's an + notation, and we can do the addition
        part1, part2 = match_data.split('+')
        value = left_sign * float(part1) + float(part2)   
    else:
        part1, part2 = match_data.split('-')  # The program does subtraction.
        value = left_sign * float(part1) - float(part2)   
    s1, s2 = add_minus_regular.split(str_expire, 1)  
    # Replace the result of the evaluation back into the expression to generate a new expression
    next_expression = "%s%s%s" % (s1, value, s2)
    return add_minus_compute(next_expression)   


# Remove the parentheses and call the methods above to compute
def remove_parentheses(str_expire):
    # If there are no parentheses in the equation, we multiply, divide, add and subtract directly.
    if len(parentheses_regular.findall(str_expire)) == 0:
        str_expire = multi_div_compute(str_expire)
        return add_minus_compute(str_expire)   
    else:
        while len(parentheses_regular.findall(str_expire)) != 0:
            brackets_str = parentheses_regular.search(str_expire).group()
            brackets_str = brackets_str.strip(r'\(([^()]+)\)')   # Remove the parentheses
            brackets_str = multi_div_compute(brackets_str)   
            brackets_str = add_minus_compute(brackets_str)   
            left_part, old_str, right_part = re.split(r'\(([^()]+)\)', str_expire, 1)  
            # Replace the result of the evaluation back into the expression to generate a new expression
            expression = left_part + brackets_str + right_part
            return remove_parentheses(expression) 
            
def main():
    # Test expression
    str_expire = "9+(-5-1.2)*2"
    result = remove_parentheses(str_expire)
    print("%s=%s" % (str_expire, result))

if __name__ == '__main__':
    main()
