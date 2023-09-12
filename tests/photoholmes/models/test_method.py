import numpy as np

from photoholmes.models.method_factory import MethodFactory

METHODS_LIST = ["naive"]

def test_image():
    for method_name in METHODS_LIST:
        method = MethodFactory.create(method_name)
        image = np.random.normal(125, 100)
        pred = method.predict(image)
        assert(image.shape == pred.shape)
        #TODO: Podríamos agregar test de tipos si queremos que las imagenes sean todo int o todo float

if __name__=='__main__':
    test_image()