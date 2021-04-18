# U-net
The reason why I put U-net here is that in pix2pix we are applying the architecture of U-net. Therefore, it is important to understand the architecture of U-net.
U-net is applied for image segmentation. The architecture consists of a contracting path tp capture context and a symmetric expanding path that enables precise localization. 
It is applied for semantic segmentation, an encoding-decoding process. 
The encoding part mainly solve the problem of 'what': we loss spatial info through this step. In order to maintain the spatial info, we are applying the encoding process, solving the problem of 'where'. 

Notice that in the original paper, the skip_connections use crop to match the size. In this approach, we are applying resize to do this. There are some small changes as well on this implementation and this implementation is applied of 2-class segmentation.
![](U-Net/u-net-architecture.png)  