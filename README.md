# <img src="https://user-images.githubusercontent.com/53294998/214486493-0840bd5f-f06b-4324-822c-74d0b056f680.png" width="30"/> Wear3Words
Convert your body measurements into simple, memorable three-word phrases! Furthermore, there's no need for a measuring tape. Input two body images - front and side view, and our app will use a machine learning model to automatically estimate your measurements. All words are intentionally scraped to have positive sentiment. 😊

<img width="1082" alt="1" src="https://user-images.githubusercontent.com/53294998/214486065-cd022c9a-92f8-4284-8d88-b8e25bb423fc.png">

## Quick Start

Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
pip install Flask numpy MarkupSafe rembg
```

Install OpenPose one directory outside of wear3words folder, follow instructions [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation).  
Get Replicate API key [here](https://replicate.com/).  
Get Livepeer API key [here](https://livepeer.org/).  

Run webapp on local host
```
cd wear3words/
flask run
```
View on displayed address

## Demo
<img width="458" alt="3" src="https://user-images.githubusercontent.com/53294998/214486280-19af920e-c8fa-48aa-af40-d7da488be938.png">

Additional option to generate an awesome video using stable diffusion, which then can be shared seamlessly via Livepeer.

<img width="485" alt="5" src="https://user-images.githubusercontent.com/53294998/214486370-d5e1c346-9cb7-4c74-9795-bc821c698d54.png">

