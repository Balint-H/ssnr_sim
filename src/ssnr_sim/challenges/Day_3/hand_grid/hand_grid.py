filename = r"../../Day_2/arm_design/solution.xml"
filename_scene = r"hand_grid_wall_scene.xml"
from dm_control import mjcf
import mujoco
import mujoco.viewer as viewer

# Modular sensing grid generation script written by Indumita Prakash


def main():
    generate_hand()


def generate_hand():
    mjcf_model = mjcf.from_path(filename)
    mjcf_model.model = "hand"
    freejoint = mjcf_model.find('joint', 'root')
    freejoint.remove()
    wall_scene = mjcf.from_path(filename_scene)
    wall_scene.model = "wall"
    pc_site = mjcf_model.find('site', 'palm_centre_site')
    add_touch_grid(pc_site, 0.02, 0.04, 0.004)
    mjcf_model = add_wall(wall_scene, mjcf_model)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_model.to_xml_string(), mjcf_model.get_assets())
    print(mjcf_model.to_xml_string())
    viewer.launch(mj_model)

class TouchGrid(object):

    def __init__(self, length, width,sens_size, margin,height):
        self.model = mjcf.RootElement()
        self.model.model = "grid"
        mid = 0.5*sens_size
        numr = int(length / (sens_size))
        numc = int(width/(sens_size))
        countr = 1
        for r in range(0, numr):
            countc = 1
            for c in range(0, numc):
                position = ((countc*mid) + (margin[0]*c) - (width+(margin[0]*(numc-1)))*0.5),0, ((countr*mid) + (margin[1]*r) - (length+(margin[1]*(numr-1)))*0.5)
                self.body = self.model.worldbody.add('body', name=f'body {c},{r}', pos=position)
                self._site = self.body.add('site', name=f'site {c},{r}', type='box', size = [ sens_size/2, height, sens_size/2])
                self.sensor = self.model.sensor.add('touch', site=self._site, name = f'sensor {c},{r}')
                self._geom = self.body.add('geom', rgba=[1, 1, 1, 0.1], name=f'geom {c},{r}', group=3, type='box', size=[sens_size/2, height, sens_size/2])
                countc += 2
            countr += 2


def add_touch_grid(site, length, width, height, sens_size=0.002, margin=(0.0015, 0.0015)):
    model = mjcf.RootElement()
    model.compiler.angle = 'degree'
    touchs_grid = TouchGrid(length=length, width=width, sens_size=sens_size, margin=margin, height=height)
    site.attach(touchs_grid.model)
    return model


def add_wall(wall_scene,mjcf_model):
    attach_site = wall_scene.worldbody.add('site', name='wall_attach_site', pos="0 0 0.1", type='sphere', size=[0.001],
                                           euler="-90 0 180", rgba = "1 1 1 0")
    attach_site.attach(mjcf_model)
    return wall_scene


if __name__ == "__main__":
    main()
