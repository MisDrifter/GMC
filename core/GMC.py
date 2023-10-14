import copy

def GMC(alpha, eta, x_cal, y_cal, h_cal, x_test, h_test, s, group_G, f=(lambda x:0), f_test = (lambda x:0), T=500, proj=None):
    '''
    eta:learning_rate
    x_cal:numpy
    y_cal:numpy
    h_cal:numpy
    x_test:numpy
    h_test:numpy
    s:mapping_function
    f:initial_function 
    group_G:list of group functions
    T:max_iteration
    return function f
    ''' 
    fx = f(x_cal)
    fx_test = f_test(x_test)
    n = x_cal.shape[0]
    for i in range(T):
        update = False
        for g in group_G:
            #if g(x)@s((fx, x, y, h))>alpha*n:
            if g(x_cal)@s(fx, x_cal, y_cal, h_cal)>alpha*n:
                update = True
                break
        if update==False:
            print(i)
            print('end')
            break
        else:
            fx = fx - eta*g(x_cal)
            fx_test = fx_test - eta*g(x_test)
            if not (proj is None):
                fx = proj(fx)
                fx_test = proj(fx_test)
    return fx, fx_test