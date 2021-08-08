# Image Quantization 1.0: Compressing Images
![screenshot](https://user-images.githubusercontent.com/25768661/128630792-cb016b4b-9fbf-48e3-a9e3-696222ddcf63.JPG)

# What is Image Quantization 1.0? 
* A C# program that maps an Image with high number of distinct colors to a really lower number of distict colors while keeping important features as much as possible, which helps in compressing the image to lower number of colors.

## Table of contents
* [Technologies](#technologies)
  * [Setup](#setup)
  * [Features](#features)
    + [Manually Input number of output distinct colors](#manually-input-number-of-output-distinct-colors)
    + [Automatically Input number of output distinct colors](#automatically-input-number-of-output-distinct-colors)

## Technologies
* Programming Language: C#
* Algorithms: Graph Theory (MST), K-means clustering.

## Setup

Graph Editor doesn't have any prerequisities rather than [.Net Framework](https://dotnet.microsoft.com/download/dotnet-framework/net48), You can run the whole project in [Visual Studio](https://visualstudio.microsoft.com/vs/) or directly run the EXE executable file in the following path

```bash
 ./ImageQuantization/bin/Debug/ImageQuantization.exe
```


## Features
### Manually Input number of output distinct colors
In Clusters textbox, you can enter the number of distinct colors that you want to map the image to and then press `Open Image` to browse and select your image.

The input image is on the right box, and the output is on the left box.

In the following example, as shown in the log cmd, the original image had appr. 8k colors and we mapped it to 4k colors keeping most of the features of the picture.

![Half ](https://user-images.githubusercontent.com/25768661/128631191-d18c6888-4bb5-40c2-afe0-9fd7725c0209.gif)




### Automatically Input number of output distinct colors
Instead of taking the number of output distinct colors as input, you can leave it blank and the program will automatically satistically detect the best number of clusters (distinct colors) to map the image to.

The input image is on the right box, and the output is on the left box.

In the following example, as shown in the log cmd, the original image had appr. 8k colors and the program found that the best number that keeps the most important features of the picture is 1k.

![Auto ](https://user-images.githubusercontent.com/25768661/128631196-203ae6ca-e14d-4606-a47d-213cdbbde937.gif)