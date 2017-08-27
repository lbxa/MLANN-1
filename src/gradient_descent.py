'''
    Framework for Linear Regression with Gradient Descent
    -----------------------------------------------------
    This micro-project contains code to transform the gradient descent 
    mathematical algorithm into a programatic example which can be applied to 
    in common Machine Learning practices to predict and input's corresponding
    output once the model has undergone supervised regression trainning.

    25.06.2017 | Lucas Barbosa | Software Design | Open Source Software (C)
'''

def cost_function(m, b, data_set):
    total_error = 0
    X = 0
    Y = 1
    for i in range(0, len(data_set)):
        x = data_set[i][X]
        y = data_set[i][Y]
        total_error += (y - ((m * x) + b))**2
    return total_error / float(len(data_set))
                        
def gradient_descent_runner(m_curr, b_curr, training_data, learning_rate):
    b_nabla = m_nabla = 0
    X = 0
    Y = 1
    N = float(len(training_data))
    for i in range(0, len(training_data)):
        x = training_data[i][X]
        y = training_data[i][Y]
        m_nabla += -(2/N) * (x * (y - ((m_curr * x) + b_curr)))
        b_nabla += -(2/N) * (y - ((m_curr * x) + b_curr)) 
    new_m = m_curr - (m_nabla * learning_rate)
    new_b = b_curr - (b_nabla * learning_rate)
    return [new_m, new_b]
                        
def gradient_descent(m_start, b_start, training_data, learning_rate, time_steps):
    m = m_start
    b = b_start
    for i in range(0, time_steps):
        m, b = gradient_descent_runner(m, b, training_data, learning_rate)
    return [m, b]
                        
if __name__ == "__main__":
    
    # --------------------- Trainning the model
    training_data = [[1, 0], [2, -1], [3, -2], [4, -3], [5, -4], [6, -5], 
    [7, -6], [8, -7], [9, -8], [10, -9], [11, -10], [12, -11], [13, -12],
    [14, -13], [15, -14], [16, -15]]
                        
    init_m = -3
    init_b = 3
                        
    new_m, new_b = gradient_descent(init_m, init_b, training_data, 0.01, 2000)
    totol_error  = cost_function(init_m, init_b, training_data)
     
    print("Initial linear model y=%sx+%s" % (init_m, init_b))
    print("Optimized linear model y=%sx+%s" % (int(new_m), int(new_b)))
    print("Total Error on the Model =", totol_error)