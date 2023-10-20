- [1. 物像关系](#1-物像关系)
- [2. FOV](#2-fov)
- [3. 相机参数](#3-相机参数)
  - [3.1. 内参 K](#31-内参-k)
  - [3.2. 外参 T](#32-外参-t)
    - [3.2.1. row/column major](#321-rowcolumn-major)
    - [3.2.2. 从点的角度理解](#322-从点的角度理解)
    - [3.2.3. 从向量的角度](#323-从向量的角度)
    - [3.2.4. w2c 和 c2w 互逆](#324-w2c-和-c2w-互逆)
- [4. 坐标系](#4-坐标系)
  - [4.1. 右手坐标系 right-handed coordinates](#41-右手坐标系-right-handed-coordinates)
  - [4.2. 各种右手的相机坐标系](#42-各种右手的相机坐标系)
- [5. project](#5-project)
  - [5.1. 相机坐标系-\>图像坐标系](#51-相机坐标系-图像坐标系)
  - [5.2. 图像坐标系-\>像素坐标系](#52-图像坐标系-像素坐标系)
  - [5.3. 综合](#53-综合)
- [6. Inverse project](#6-inverse-project)

---
## 1. 物像关系

![图 16](../../images/b3907e4ae3b5cf7e2ad5b2f5a29ad448f277b5479e20c1b95cd299c2ce656126.png)  

![图 17](../../images/df934186093fafd9f9590af9a0e32b5c441cceda3cedc23f6ab242324051a987.png)  


$\dfrac{1}{f} = \dfrac{1}{z_0} + \dfrac{1}{z_i}$

f为焦距， $z_0$为物距（物体到透镜的距离）， $z_i$为像距（透镜到像面的距离）。


[由此可以推出不同的成像关系](https://www.zhihu.com/question/38929736/answer/2327108553)

## 2. FOV
视场 Field of view（FOV），是一个角度。

> 当传感器大小固定时，焦距越短，FOV越大；焦距越长，FOV越小。

![图 11](../../images/ee747b21ed49adc9a8a76c96dfca5e6906b09dfed5bb16818d20f47af7f91a6d.png)

> 当焦距固定时，传感器大小越小，FOV越小。

![图 14](../../images/cb9e61b95bb327af9712617afd7ac0de986308aeb7bc522376dddd29d8235b98.png)  


> 公式就是tan三角函数联系起焦距与高宽、或物距与物高宽。

![图 1](../../images/42233c45b2aac9127c39b30870163a7a6cf81a6e7bb3da56bc8c1d73973a5155.png)  


![图 1](../../images/2db836f5a66a8f1a2111baafb5742a7a286a6c8225ec8e6db47c299d391556fc.png)  

W是图像的宽度（传感器大小），$\omega$是x轴的视角宽度，f是焦距。
```python
# `camera_angle_x`即在x轴的视角，对应图像的宽度。
focal = .5 * W / np.tan(.5 * camera_angle_x)
```
`"camera_angle_x": 0.5235987755982988,` 就是比如30°， Π/6 = 0.5235987755982988

`W`的单位是pixel， f的单位也是pixel。


> 镜头参数

![图 12](../../images/53da8b7cab4fd935b085033322fe5dd77aa8b274a90dabff368d2862238f5b09.png)  

我们以35mm-format的底片大小为标准，17mm、50mm、200mm、28mm的焦距是在这样大小的底片上，这个焦距是等效的虚指。意思是，实际手机镜头的焦距很小，对应的也是很小的底片大小。

拍照的效果就是，视场越窄，镜头拍到的就越远。

![图 13](../../images/e2954d730381f66ae2b5dbef7c487b20f207d99d59d28ed24d7a42afe1fe965c.png)  

> maintain same FOV

![图 15](../../images/0cb82c87d007dc87a7a461bd68515b29d8bd636eaf55de47619228a3af050c1c.png)  


## 3. 相机参数

相机两个参数：内参和外参

- 外参extrinsics，即**T**
    描述相机的位姿（位置t是指相机在空间中的哪个地方，而姿态R则是指相机的朝向）

- 内参intrinsics，即**K**
    固定不变的，如果不知道可以通过**标定**求得。

- 内参共有，外参变化：
    由于多组图片都是同一个相机拍摄得到，所以其内参数由多组图像共有，而外参数随着不同的图像将发生变化
- 都用齐次坐标表示
  

### 3.1. 内参 K

透射：The intrinsic matrix transforms 3D camera cooordinates to 2D homogeneous image coordinates.

$K = \begin{bmatrix} \alpha f_x & s & c_x\\ 0 & \beta f_y & c_y\\ 0 & 0 & 1\end{bmatrix}$

![图 28](../../images/c0e0f29877a99b74141b06fbb558c3df63057f9f80a95dcd53abb91ab9876c7c.png)  

平移操作放在最后（即最左）。

在内参矩阵中还有个参数 $s$（通常也可当成0），用来建模像素是平行四边形而不是矩形，与像素坐标系的u，v轴之间的夹角$\theta$的正切值$tan(\theta)$成反比，因此当 $s = 0$时，表示像素为矩形。

```python
# 缩放尺寸为1，不缩放
# 平移到图像中心
K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]
])
```

### 3.2. 外参 T

let's consider **translation for points (positions)** and **rotations for vectors (directions)**.

> 相机外参是一个4x4的矩阵。

相机外参叫做**world-to-camera (w2c)矩阵**，其作用是把3D世界坐标系的坐标变换到3D相机坐标系的坐标。

相机外参的逆矩阵被称为**camera-to-world (c2w)矩阵**，其作用是把3D相机坐标系的坐标变换到3D世界坐标系的坐标。


#### 3.2.1. row/column major

red: x-axis, green: y-axis, blue: z-axis

| Column-Major Vector| Row-Major Vector|
|:-:|:-:|
|从右到左, Post-multiplication | 从左到右, Pre-multiplication |
|$P^{\prime} = T_3*T_2*T_1*P$ | $P^{\prime}=P*T_1*T_2*T_3$ |
| API: OpenGL, Blender, PBRT|API: Direct X, Maya |
|${ \begin{bmatrix} \color{red}{X0}& \color{green}{Y0}&\color{blue}{Z0}&X\\ \color{red}{X1}& \color{green}{Y1}&\color{blue}{Z1}&Y\\ \color{red}{X2}& \color{green}{Y2}&\color{blue}{Z2}&Z\\0&0&0&1\end{bmatrix} } = \begin{bmatrix} \color{red}{\textbf{Col}_X} & \color{green}{\textbf{Col}_Y} & \color{blue}{\textbf{Col}_Z} & \textbf{Col}_t\\ 0 &0 & 0 & 1\end{bmatrix}$ | ${\begin{bmatrix} \color{red}{X0}& \color{red}{X1}&\color{red}{X2}&0\\ \color{green}{Y0}& \color{green}{Y1}&\color{green}{Y2}&0\\ \color{blue}{Z0}& \color{blue}{Z1}&\color{blue}{Z2}&0\\ X & Y &Z & 1 \end{bmatrix} } = \begin{bmatrix} \color{red}{\textbf{Row}_X} & 0\\ \color{green}{\textbf{Row}_Y} & 0\\ \color{blue}{\textbf{Row}_Z} & 0 \\ \textbf{Row}_t & 1\end{bmatrix}$ |

**NeRF主要使用 Column-Major 的 c2w**

属于刚体变换，包括旋转和平移操作（先平移后旋转）。

比如column-major w2c：
- 世界坐标系的欧式点$P_{w}=[X_{w}, Y_{w}, Z_{w}]^\top$，相机坐标系的欧式点$P_{c}=[X_{c}, Y_{c}, Z_{c}]^\top$，

    $$\begin{aligned}
    P_{c}&=RP_{w}+t \\
    \begin{bmatrix} X_{c} \\ Y_{c} \\ Z_{c}  \end{bmatrix}  
    &= R \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w}  \end{bmatrix} + \begin{bmatrix} t_{x} \\  t_{y} \\ t_{z}  \end{bmatrix}
    \end{aligned}$$

- 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]^\top$，相机坐标系的欧式点$P_{c}=[X_{c}, Y_{c}, Z_{c}]^\top$，

    $$\begin{aligned}
    P_{c}&=\begin{bmatrix} R & t \end{bmatrix}P_{w}\\
    \begin{bmatrix} X_{c} \\ Y_{c} \\ Z_{c} \end{bmatrix}  
    &= \begin{bmatrix} R & t \end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \end{aligned}$$

- 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]^\top$，相机坐标系的齐次坐标点$P_{c}=[X_{c}, Y_{c}, Z_{c}, 1]^\top$，
    $$\begin{aligned}
    P_{c}&=\begin{bmatrix} R & t \\ 0^T & 1 \end{bmatrix}P_{w}\\
    \begin{bmatrix} X_{c} \\ Y_{c} \\ Z_{c} \\ 1 \end{bmatrix}  
    &= \begin{bmatrix} R & t \\ 0^T & 1  \end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \end{aligned}$$

    甚至可以进一步分解，这样就很明显是先乘旋转矩阵，后乘平移矩阵。

    $$
    \begin{aligned}
    \left[\begin{array}{c|c}R&\mathbf{t}\\\hline\mathbf{0\top}&1\end{array}\right]
    & =\left[\begin{array}{c|c}I&\mathbf{t}\\\hline\mathbf{0\top}&1\end{array}\right]\times\left[\begin{array}{c|c}R&\mathbf{0}\\\hline\mathbf{0\top}&1\end{array}\right]  \\
    &=\left[\begin{array}{ccc|c}1&0&0&t_1\\0&1&0&t_2\\0&0&1&t_3\\\hline0&0&0&1\end{array}\right]\times\left[\begin{array}{ccc|c}r_{1,1}&r_{1,2}&r_{1,3}&0\\r_{2,1}&r_{2,2}&r_{2,3}&0\\r_{3,1}&r_{3,2}&r_{3,3}&0\\\hline0&0&0&1\end{array}\right]
    \end{aligned}
    $$

比如row-major w2c：
- 世界坐标系的欧式点$P_{w}=[X_{w}, Y_{w}, Z_{w}]$，相机坐标系的欧式点$P_{c}=[X_{c}, Y_{c}, Z_{c}]$，

    $$\begin{aligned}
    P_{c}&=P_{w}R+t \\
    \begin{bmatrix} X_{c} & Y_{c} & Z_{c}  \end{bmatrix}  
    &= \begin{bmatrix} X_{w} & Y_{w} & Z_{w}  \end{bmatrix} R + \begin{bmatrix} t_{x} & t_{y} & t_{z} \end{bmatrix}
    \end{aligned}$$

- 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]$，相机坐标系的欧式点$P_{c}=[X_{c}, Y_{c}, Z_{c}]$，

    $$\begin{aligned}
    P_{c}&=P_{w}\begin{bmatrix} R \\ t \end{bmatrix}\\
    \begin{bmatrix} X_{c} & Y_{c} & Z_{c} \end{bmatrix}  
    &= \begin{bmatrix} X_{w} &  Y_{w} & Z_{w} & 1 \end{bmatrix} \begin{bmatrix} R \\ t \end{bmatrix}
    \end{aligned}$$

- 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]$，相机坐标系的齐次坐标点$P_{c}=[X_{c}, Y_{c}, Z_{c}, 1]$，
    $$\begin{aligned}
    P_{c}&=\begin{bmatrix} R &  0 \\ t & 1 \end{bmatrix}P_{w}\\
    \begin{bmatrix} X_{c} & Y_{c} & Z_{c} & 1 \end{bmatrix}  
    &= \begin{bmatrix} X_{w} & Y_{w} & Z_{w} & 1 \end{bmatrix}\begin{bmatrix} R &  0 \\ t & 1 \end{bmatrix}  
    \end{aligned}$$

    甚至可以进一步分解，这样就很明显是先乘旋转矩阵，后乘平移矩阵。

    $$
    \begin{aligned}
    \left[\begin{array}{c|c}R&\mathbf{0}\\\hline\mathbf{t}&1\end{array}\right]
    & =\left[\begin{array}{c|c}R&\mathbf{0}\\\hline\mathbf{0}&1\end{array}\right]\times \left[\begin{array}{c|c}I&\mathbf{0}\\\hline\mathbf{t}&1\end{array}\right] \\
    &=\left[\begin{array}{ccc|c}r_{1,1}&r_{1,2}&r_{1,3}&0\\r_{2,1}&r_{2,2}&r_{2,3}&0\\r_{3,1}&r_{3,2}&r_{3,3}&0\\\hline0&0&0&1\end{array}\right] \times \left[\begin{array}{ccc|c}1&0&0&0\\0&1&0&0\\0&0&1&0\\\hline t_1&t_2&t_3&1\end{array}\right]
    \end{aligned}
    $$

#### 3.2.2. 从点的角度理解

c2w的含义: camera's pose matrix in world coordinate.

c2w矩阵是一个4x4的矩阵，左上角3x3是旋转矩阵R，又上角的3x1向量是平移向量T。有时写的时候可以忽略最后一行[0,0,0,1]。

$$\begin{bmatrix}R&t\\0^\top &1\end{bmatrix}=\begin{bmatrix}\begin{array}{ccc|c}r_{11}&r_{12}&r_{13}&t_1\\r_{21}&r_{22}&r_{23}&t_2\\r_{31}&r_{32}&r_{33}&t_3\\\hline0&0&0&1\end{array}\end{bmatrix}$$

$$\begin{bmatrix}R&t\end{bmatrix}=\begin{bmatrix}\begin{array}{ccc|c}r_{11}&r_{12}&r_{13}&t_1\\r_{21}&r_{22}&r_{23}&t_2\\r_{31}&r_{32}&r_{33}&t_3\end{array}\end{bmatrix}$$



The camera's extrinsic matrix describes the camera's location in the world $\mathbf{t}$, and what direction it's pointing $\mathbf{R}$. 

For a **Column-Major** transform matrix, the first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the forth column X, Y, Z values define the camera origin. 具体来说, camera orientation 是当前坐标系的轴在要变换到的另一坐标系的轴的方向, camera origin 是当前坐标系的原点在要变换到的另一坐标系的下的坐标。比如，c2w，则旋转矩阵的每一列分别表示了相机坐标系的XYZ轴方向在世界坐标系下对应的XYZ轴方向，平移向量表示的是相机坐标系的原点在世界坐标系中的位置。

- T
    
    $\mathbf{T} \in SE(3)$
    
    $SE(n) = \left\{\mathbf{T}=\begin{bmatrix}\mathbf{R} & \mathbf{t}\\ 0^\top & 1\end{bmatrix} \in \R^{4\times 4}|\mathbf{R}\in SO(3),\mathbf{t}\in \R^3\right\}$

- R
    
    $\mathbf{R} \in SO(3)$
    
    $SO(n) = \{\mathbf{R} \in \R^{n\times n}|\mathbf{R}\mathbf{R}^T=I,\det(\mathbf{R})=1\}$
    - 旋转矩阵是一个正交矩阵（正交矩阵的逆等于其转置矩阵，则有$\mathbf{R}^{-1} = \mathbf{R}^{\top}$）
    - 行列式值为1

- t

    $\mathbf{t}\in \R^3$



#### 3.2.3. 从向量的角度

![图 5](../../images/ee3d0db691dffa4d96f9ffcaabf9cb0e52ac6a3ded1da48014924c67fb1d696f.png)  

描述点B。在绿色坐标系下，B点(1,2)。在蓝色坐标系下，B点(2,2)。怎么转化？借助向量。

描述向量AB。在绿色坐标系下，AB是起点(0,0)和方向向量(1,2)，即AB(1,2)=(0,0)+(1,2)。在蓝色坐标系下是CB=CA+AB, (2,2)=(0,1)+(2,1)。

也即A点(0,1)和B点(2,2)=(1,2)-(-1,0)。

怎么做到从绿色到蓝色？旋转坐标系，方向向量(2,1)变化为(1,2)，平移向量(-1,0)就是在绿色坐标系下观察的世界坐标系原点的位置。

![图 6](../../images/a3b6257693f7e85d96f84d846a740dac3f521287df3779978b9079484f8d3203.png)  

相机坐标系虚线坐标轴，世界坐标系彩色坐标轴。相机坐标的黑色OA，选转后世界坐标的OB，在相机坐标下看世界坐标原点的平移量是粉色的OO'，世界坐标的O'C = OB - OO'。

也就是说，关键点，**世界坐标下的向量 = 旋转后的向量 - 相机坐标下看世界坐标原点的平移向量**，或者，****世界坐标下的向量 = 旋转后的向量 + 世界坐标下看相机坐标原点的平移向量****。后者才是矩阵中的 $t$。

> 获取

colmap已经设定好了世界坐标系，外参也是依据此世界坐标系的。

![图 1](../../images/962653656f477ecdb5999f0d3a9108a94d3d07c8a53af8de76055e3d2368f298.png)  


#### 3.2.4. w2c 和 c2w 互逆


> column-major w2c↔c2w，两矩阵互逆。

$$
\begin{aligned}
\left[\begin{array}{c|c}\mathbf{R_c}&\mathbf{C}\\\hline \mathbf{0}\top&1\end{array}\right]
& = \left[\begin{array}{c|c}\mathbf{R}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}  \\
&=\left[\left[\begin{array}{c|c}\mathbf{I}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]\left[\begin{array}{c|c}\mathbf{R}&0\\\hline\mathbf{0}&1\end{array}\right]\right]^{-1}& (\text{decomposing rigid transform})  \\
&=\left[\begin{array}{c|c}\mathbf{R}&0\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}\left[\begin{array}{c|c}\mathbf{I}&\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]^{-1}& (\text{distributing the inverse})  \\
&=\left[\begin{array}{c|c}\mathbf{R}^\top &0\\\hline\mathbf{0}\top&1\end{array}\right]\left[\begin{array}{c|c}\mathbf{I}&-\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]& \text{(applying the inverse)}  \\
&=\left[\begin{array}{c|c}\mathbf{R}^\top&-\mathbf{R}^\top\mathbf{t}\\\hline\mathbf{0}\top&1\end{array}\right]& (\text{matrix multiplication}) 
\end{aligned}
$$


即 $T_{w2c} = [\mathbf{R}, \mathbf{t}], 则T_{c2w} = T_{w2c}^{-1} = [\mathbf{R}^\top, -\mathbf{R}^\top\mathbf{t}]$


## 4. 坐标系
坐标系 Coordinate Frames

![图 4](../../images/f543774d2d57cb991abf472608bf10221a356f9e287b95aad9a49a5d7548670f.png)  

Express motion information relative to a reference frame. 

参考坐标系 reference frame可以分为：
- Camera Frame
- World Frame

之后的图像坐标系和像素坐标系只有xy轴，且和相机坐标系的xy轴方向保持一致。
### 4.1. 右手坐标系 right-handed coordinates

手掌：用右手的**4个指头从a转向b**（合拳，而不是松拳），大拇指朝向就是aXb的方向。

三指：右手，大拇指a，食指b，中指的方向就是axb。（是大食中、食中大、中大食的升序，而不是中食大等的降序）

![图 2](../../images/e64bdd1708f9aaa11f38fd1efd544325dac5aee40d904accfee17c096283c59e.png)  


> 将左右手性和right-up-forward联系在一起，而不是xyz

![图 3](../../images/e8c52001b6f627664240997c2677db5bb989c04d4ada4e4dcaa433de12a624af.png)  

图中b还是右手性，是认为up是z轴，按照right-up-forward来判断它还是右手性。

The only thing that defines the handedness of the coordinate system is the orientation of the left (or right) vector relative to the up and forward vectors, regardless of what these axes represent.

Currently, the industry standard tends to be the right-hand XYZ coordinate system where x points to the right, y is up, and z is backwards (coming out of the screen).
### 4.2. 各种右手的相机坐标系



![图 8](../../images/0b2e24a1c6d97650f49d5d02e08f0f244fa60e9a700c62330261ddb30daeb61d.png)  

We use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction.


![图 3](../../images/d7811eeb810841979e5f8cbd88f6e6d71e2744c4d464081863dd8d93079e2370.png)  


![图 4](../../images/48b46404b809bc67b121791f53db54e68eb85eaec2c3968048c70024f1e81280.png)  
![图 5](../../images/872d8ed4d1075b1b5cadc6434a6987e6d28d73e50a72fcdb8dc49a8cb743a7c6.png)  

或者
```python
# 补全成4x4的矩阵
# flip the sign of the y-axis (from down to up) and z-axis (from forwards to backwards)
camtoworlds_opengl = camtoworlds_opencv @ np.diag([1, -1, -1, 1])
```

## 5. project

图像的成像过程经历了世界坐标系—>相机坐标系—>图像坐标系—>像素坐标系这四个坐标系的转换，如下图所示：

![图 21](../../images/080cbdb99e930d58e34b77462b1875009c9f9e42dc825a8dbc61d6d2296b527b.png)  

- 3D世界坐标系 world coordinate：是三维世界的绝对坐标系，我们需要用它来描述三维环境中的任何物体的位置，用 $(x_{w}, y_{w},z_{w})$ 表示其坐标值。
 
- 3D相机坐标系 camera coordinate：以相机光心为原点的坐标系，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，相机的光轴为Z 轴，具体方向看相机坐标系是用的哪种。用 $(x_{c}, y_{c},z_{c})$ 表示其坐标值。

- 2D图像坐标系 image coordinate：在 image plane上，即以 principal point 为原点的坐标系，X轴和Y轴的方向同相机坐标系，用 $(x, y)$ 表示其坐标值。图像坐标系是用物理单位表示像素在图像中的位置。

- 2D像素坐标系 pixels coordinate：以图像平面左上角为原点的坐标系 ，X轴和Y轴的方向同图像坐标系，用 $(u,v)$ 表示其坐标值。像素坐标系就是以像素为单位的图像坐标系。

### 5.1. 相机坐标系->图像坐标系

![图 18](../../images/cdb55d1e17a03b6ffc22e9e3dad8fc263bb6d8bc33cc849254893960c4b1e2c6.png)  

图像坐标系的x和y轴方向和相机坐标系的保持一致。

![图 22](../../images/29f0b1870a1da2e77cd3ad296e1e79b07bbc79296d33644be2e64ee34b297e79.png)  

从3D的相机坐标系下的欧式点 $(X_{c}, Y_{c}, Z_{c})$ 到2D的图像坐标系下的欧式点 $(x,y)$  

The focal length is the distance between the pinhole and image plane. 
$$\begin{aligned}
\dfrac{f}{Z_{c}} &= \dfrac{x}{X_c} = \dfrac{y}{Y_c}
\\ x&=f\dfrac{X_{c}}{Z_{c}}
\\ y&=f\dfrac{Y_{c}}{Z_{c}}\end{aligned}$$

$\begin{bmatrix} f_x & 0 & 0 & 0\\ 0 & f_y & 0 & 0\\ 0 & 0 & 1 & 0\end{bmatrix}  \begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c} \\ 1 \end{bmatrix}
=\begin{bmatrix} f_xX_c \\ f_yY_c \\ Z_c \end{bmatrix} 
=Z_c\begin{bmatrix} f_x\dfrac{X_{c}}{Z_{c}} \\ f_y\dfrac{Y_{c}}{Z_{c}} \\ 1 \end{bmatrix}
=Z_c\begin{bmatrix}x \\y \\1 \end{bmatrix}$

$\begin{bmatrix} f_x & 0 & 0\\ 0 & f_y & 0\\ 0 & 0 & 1\end{bmatrix}  \begin{bmatrix} X_{c} \\  Y_{c} \\ Z_{c}\end{bmatrix}
=\begin{bmatrix} f_xX_c \\ f_yY_c \\ Z_c \end{bmatrix} 
=Z_c\begin{bmatrix} f_x\dfrac{X_{c}}{Z_{c}} \\ f_y\dfrac{Y_{c}}{Z_{c}} \\ 1 \end{bmatrix}
=Z_c\begin{bmatrix}x \\y \\1 \end{bmatrix}$


PS：倒像问题 / image plane

the image plane that is perpendicular to the optical axis of the camera and at a fixed distance from the pinhole (the focal length).

![图 23](../../images/2d8672f8aa24c34bc1097463b893585081be453ffc927c33020f1f21f5d99d43.png)  

P点的x是负坐标，P'点的x是正坐标。

$$\dfrac{f}{Z_{c}} = -\dfrac{x}{X_c} = -\dfrac{y}{Y_C}$$

其中负号表示成的像是倒立的。为了简化模型，我们可以把成像平面对称到相机前方，和三维空间点一起放在摄像机坐标系的同一侧，这样做可以把公式中的负号去掉，使式子更加简洁。

![图 24](../../images/59a8c83face6fedb26b184273bccb9433d009275d722a72dae996a5f958c46e7.png)  

### 5.2. 图像坐标系->像素坐标系

![图 27](../../images/d2019bb8a07eeb32e230a28112f6751b5022826038a05d059debcf03c79defa4.png)  


像素坐标系：以左上角点为原点，u轴向右与x轴平行，v轴向右与y轴平行。像素坐标系和图像坐标系之间，相差了一个缩放 $\alpha, \beta$和原点的平移 $c_x, c_y$。

$$\begin{aligned}
u&=\alpha x + c_x
\\ v&=\beta y + c_y
\end{aligned}$$

$
Z_c \begin{bmatrix}\alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix}x \\y \\1 \end{bmatrix} = 
Z_c\begin{bmatrix} u \\v \\ 1\end{bmatrix}
$

### 5.3. 综合

![图 26](../../images/b899ce078ef1a11a8bdc6fdde427448eaecbada3eb4ffa9557a90a3afac8dd66.png) 


像素坐标的齐次坐标点 $P_{uv}=[u, v]$. 三种运算方式：

1. 加法
   
    $$ Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} = K\left( R\begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + t \right)$$

2. 世界坐标系的欧式点$P_{w}=[X_{w}, Y_{w}, Z_{w}]$，像素坐标的齐次坐标点 $P_{uv}=[u, v]$

    $$\begin{aligned}
    Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} 
    &= \begin{bmatrix} \alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix}
    \begin{bmatrix} f_x & 0 & 0\\ 0 & f_y & 0\\ 0 & 0 & 1\end{bmatrix}
    \begin{bmatrix} R & t\end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= 
    \begin{bmatrix} \alpha f_x & 0 & c_x\\ 0 & \beta f_y & c_y\\ 0 & 0 & 1\end{bmatrix}
    \begin{bmatrix} R & t\end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= KTP_w
    \end{aligned}
    $$


1. 世界坐标系的齐次坐标点$P_{w}=[X_{w}, Y_{w}, Z_{w}, 1]

    $$\begin{aligned}
    Z_c\begin{bmatrix} u \\ v \\ 1\end{bmatrix} 
    &= \begin{bmatrix} \alpha & 0 & c_x \\0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix}
    \begin{bmatrix} f_x & 0 & 0 & 0\\ 0 & f_y & 0 & 0\\ 0 & 0 & 1 & 0\end{bmatrix}
    \begin{bmatrix} R & t \\ 0^T & 1  \end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= 
    \begin{bmatrix} \alpha f_x & 0 & c_x & 0\\ 0 & \beta f_y & c_y & 0\\ 0 & 0 & 1 & 0\end{bmatrix}
    \begin{bmatrix} R & t \\ 0^T & 1  \end{bmatrix}  \begin{bmatrix} X_{w} \\  Y_{w} \\ Z_{w} \\ 1 \end{bmatrix}
    \\ &= KTP_w
    \end{aligned}
    $$

**相机深度**$z_{c}$ 乘以 **像素坐标**$P_{uv}=\begin{bmatrix} u \\ v \\ 1\end{bmatrix}$ 等于 **相机内参**K 乘以 **相机外参T** 乘以 **世界坐标**$P_{w}$

像素坐标系下的一点可以被认为是三维空间中的一条射线， $z_{c}$ 就是像素点在相机坐标系下的深度。

## 6. Inverse project

> 例子

例子：已知，$T_{c2w} = [\mathbf{R}, \mathbf{t}]$， $P_c=[u,v,1]^\top$

则，
$$P_w = \mathbf{R}\mathbf{K}^{−1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} + \mathbf{t} $$

- a target pixel $x\in\mathbf{RP}^{2}$ , c2w extrinsics $[R | t]$ , intrinsics $K$ .
- ray origin $o=\mathbf{t}$, ray direction $r=\mathbf{R}\mathbf{K}^{−1}[u,v,1]^\top$

例子：已知，$T_{w2c} = [\mathbf{R}, \mathbf{t}]$， $P_c=[u,v,1]^\top$

则，

$$P_w = \mathbf{R}^{\top}\mathbf{K}^{−1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} + (-\mathbf{R}^\top\mathbf{t})$$

- a target pixel $x\in\mathbf{RP}^{2}$ , w2c extrinsics $[R | t]$ , intrinsics $K$
- ray origin $o=-\mathbf{R}^\top\mathbf{t}$, ray direction $r=\mathbf{R}^{\top}\mathbf{K}^{−1}[u,v,1]^\top$