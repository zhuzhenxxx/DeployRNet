# DeployRNet
由于模型和依赖库太大，暂时没上传，手动把文件拷过来。

把models文件夹拷贝至当前目录，存放模型
```
  models
    -FaceCrop.onnx
```
sharedlib文件夹拷贝至当前目录
```
  sharedlib
    -onnxruntime-win-x64-1.13.1
    -opencv
```

测试代码处理图片以Mat对象为单位或者是图片写到文件以文件路径为单位。

当前我们采用opencv的Mat对象作为图片处理整个流程的载体。

推理的时候还是传入图片路径，由推理模块内部进行加载。推理出的结果为opencv的Mat对象。

各种后处理方法封装在FaceProcess类中。输入Mat对象，输出为Mat对象。以此来实现链式操作。


TODO:

后续可将结果缓存在类中，实现链式的连续调用。 

目前只修改了人脸分割的返回值。其他函数还未改完。
