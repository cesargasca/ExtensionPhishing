
# Detección de correos de phishing

## Para agregar a tu navegador chrome
Primero descarga los archivos
    * En las extensiones de chrome habilita el modo desarrollador
    * Agrega la carpeta "extension-chrome"
    * Entra a mail.google.com y abre un correo
En el cuerpo del correo debe mostrarse una leyenda: PHISHING en rojo o NO PASA NADA en verde

## Para hacer modificaciones al código
    * Descarga el repositorio
    * En la carpeta de entrenamiento esta entrenamiento.py y los archivos necesarios para funcionar
        * Para ejecutar: python3 entrenamiento.py
    * Para modificar la extensión, pudes hacerlo desde src/extension.js y seguir las siguientes instrucciones:
    
````
cd extension-chrome
npm install
npm update
npm run build
````

Suerte!
