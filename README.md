# Depth-vOICe

Depth-vOICe is a revitalized version of vOICe, a groundbreaking synesthesia device designed to empower the visually impaired. This project aimed to enhance the device architecture, address existing issues, and improve user experience by introducing depth into audio signals and overcoming arbitrary lighting challenges.

### Features
- Depth Integration: Implemented depth perception using MiDaS, allowing users to perceive the spatial layout of monocular videos through sound which
  1. Mitigates issues of the original program caused due to differences in scene lighting
  2. Allows users to control the range of interest through expanded settings
- Object Detection: Users can filter objects of interest which results in reduced noise and highly focused soundscapes

In the following example, we see noise-free signals highlighting major objects of interests i.e. the bus and car. The advancing car is identifiable as the portion of the signal that increases in amplitude and shifts relative to the vehicle.

![](https://github.com/statix531441/Depth-vOICe/blob/main/outputs/car2cam/video.gif)

### To Run the Program

```
git clone https://github.com/statix531441/Depth-vOICe
cd Depth-vOICe
pip install -r requirements.txt
python convert.py --video <path-to-video>
```
