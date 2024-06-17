import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.svm import NuSVR
import pickle

def collect_sensor_data(model, data, z_range, y_range,):
    sensor_data = list()
    for z in z_range:
        for y in y_range:
            sensor = shoulder_to_sensor(model, data, z, y)
            sensor_data.append([z, y, *sensor])
    return np.array(sensor_data)


def shoulder_to_sensor(model, data, shoulder_z, shoulder_y):
    data.joint("shoulder_z").qpos = shoulder_z
    data.joint("shoulder_y").qpos = shoulder_y
    mujoco.mj_forward(model, data)
    return data.sensordata


def main():
    model = mujoco.MjModel.from_xml_path(filename='../../../xml/05_nonlinear_sensing.xml')
    data = mujoco.MjData(model)
    resolution = 50
    z_range = np.linspace(*model.joint("shoulder_z").range, resolution)
    y_range = np.linspace(*model.joint("shoulder_y").range, resolution)
    sensor_data = collect_sensor_data(model, data, z_range, y_range)

    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

    for i in range(model.nsensor):
        ax = axs.flatten()[i]
        X = z_range
        Y = y_range
        X, Y = np.meshgrid(X, Y)
        Z = np.reshape(sensor_data[:, 2+i], (resolution, resolution))
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    plt.show()

    sv_z = NuSVR(kernel='rbf', C=1, nu=0.8).fit(sensor_data[:, 2:], sensor_data[:, 0])
    sv_y = NuSVR(kernel='rbf', C=1, nu=0.8).fit(sensor_data[:, 2:], sensor_data[:, 1])
    with open('sv_z.pickle', 'wb') as handle:
        pickle.dump(sv_z, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sv_y.pickle', 'wb') as handle:
        pickle.dump(sv_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass


if __name__ == '__main__':
    main()

