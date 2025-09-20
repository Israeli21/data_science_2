import numpy as np
import pickle as pk


np.random.seed(np.random.randint(1000))

#genration interval
dt = 0.001
#interval of choosing a point for data
st = 100
#length of trajectory
T = 1000
#number of warming points 
warm = 1000

def lorenz(x, y, z, s=10, r=28, b=8/3):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def generation(dt, T, st, warm):
    num_dots = int(T/dt/st + warm)

    xs = np.empty(num_dots)
    ys = np.empty(num_dots)
    zs = np.empty(num_dots)
    


    xs[0] = np.random.uniform(-20,20)
    ys[0] = np.random.uniform(-30,30)
    zs[0] = np.random.uniform(0,50)
    

    for i in range(1, num_dots):
        if i == num_dots * 0.75:
            print('75%')
        elif i == num_dots * 0.5:
            print('50%')
        elif i == num_dots * 0.25:
            print('25%')


        x_dot, y_dot, z_dot = lorenz(xs[i-1], ys[i-1], zs[i-1])

        xs[i] = xs[i - 1] + (x_dot * dt)
        ys[i] = ys[i - 1] + (y_dot * dt)
        zs[i] = zs[i - 1] + (z_dot * dt)

        for o in range(1,st):
            
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])

            xs[i] += (x_dot * dt)
            ys[i] += (y_dot * dt)
            zs[i] += (z_dot * dt)


    return xs[warm:], ys[warm:], zs[warm:]



data = np.array(generation(dt, T, st, warm)).T


with open('trajectory.pkl', 'wb') as file:
    pk.dump(data, file)