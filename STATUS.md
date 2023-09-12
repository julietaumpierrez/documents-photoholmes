Archivo nuestro para comentar cambios y cosas pendientes.

# ToDo
 - Arreglar que las cosas corran como photoholmes como root, por defecto. El estado actual es que solo scipts en 'src/' funcan. A causa de esto, no creo que funque ningún pytest por ahora.
 - Implementar métodos. Para hacerlo, guiarse por la estructura de Naive, es importante que herede de la clase Method. Consideraciones:
    * El core está en overridear el método predict, y si es necesario también predict_mask. Este último podría utilizar cosas de post-processing, o podemos hacerlo en otro lado como gusten.
    * La idea es que se inicialice siempre desde el config, con lo cual no hay que darle tanta bola al init  y sobre todo no poner ahí los parámetros por defecto. El config es una manera de evitar tener constantes por todos lados.
    * El notebook 09-11 hace una prueba muy básica del método. Puede estar bueno como primer test de si es coherente la implementación del método.
    * Una vez hecho: 
        1. Agregar a la lista de '__init__.py' en models
        2. Agregar a la method_factory, tanto el tipo como la conversión.
        3. Correr pytest, fijarse si se puede agregar alguno. Por boludo que sea, lo vamos llenando

- Implementar métricas y metodología de evaluación.

# Comentarios:
 - 