#include "NvCodecRender.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

#include <GLES3/gl31.h>
#include "ContextEGL.h"
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";
const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform float inColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(inColor, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

unsigned int shaderProgram = 0;
unsigned int vertexShader = 0;
unsigned int fragmentShader = 0;
unsigned int VBO, VAO,FBO,texId0,texId1;

ContextEGL glContext;

void BindTexture(int texId){
    glBindFramebuffer(GL_FRAMEBUFFER, FBO); // Bind custom framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
}

void InitRenderTexture(int width,int height){
    // Step 1: Create FBO
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);

    // Step 2: Create Texture 0
    glGenTextures(1, &texId0);
    glBindTexture(GL_TEXTURE_2D, texId0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Step 3: Attach Texture to FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId0, 0);


    // Step 2: Create Texture 0
    glGenTextures(1, &texId1);
    glBindTexture(GL_TEXTURE_2D, texId1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Step 3: Attach Texture to FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId1, 0);

    // Step 4: Check FBO status
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        // Handle errors here
    }

}

void InitRender(){
    glContext.Initialize();

    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    // fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
 
 
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float vertices[] = {
        -1.0f, -1.0f, 0.0f, // left
         1.0f, -1.0f, 0.0f, // right
         0.0f,  1.0f, 0.0f  // top
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

   
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

float testColor = 0.0;
void DoRender(){
    glViewport(0, 0, glContext.GetWindowWidth(), glContext.GetWindowHiehgt()); 

    GLint currentFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFBO);
    
    GLint readBuffer;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readBuffer);
    
    GLint drawBuffer;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawBuffer);
    
    // printf("current fbo --------- : %d" ,currentFBO);

    glClearColor(0.5f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // draw our first triangle
    glUseProgram(shaderProgram);
    auto uInColor = glGetUniformLocation(shaderProgram, "inColor");
    testColor += 0.01;
    glUniform1f(uInColor,std::sin(testColor));

    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
    glDrawArrays(GL_TRIANGLES, 0, 3);    
}

bool bwrite = true;
void ReadPixel(int width,int height){
    uint32_t bufferSize = width * height * 4;
    uint8_t buf[bufferSize];
    glReadPixels(0, 0, width, height, GL_RGBA,GL_UNSIGNED_BYTE,buf);
    // printf("ReadPixels: %d,%d,%d,%d;%d,%d,%d,%d\n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5],buf[6], buf[7]);

    if( bwrite ){
        std::string path = "test.png";
        void* data = static_cast<void*>(buf);
        stbi_flip_vertically_on_write(true);
        stbi_write_png(path.c_str(), width, height, 4, data, width * 4);
        bwrite = false;
    }

}

static CUcontext cuContext = NULL;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

//假设你使用标准的 GL_RGBA 格式和 GL_UNSIGNED_BYTE 类型上传纹理，pixelType 可以定义成：
// typedef unsigned char pixelType; // 定义像素类型，可以根据需求选择不同的类型，如 float 或 unsigned char
typedef float pixelType; // 定义像素类型，可以根据需求选择不同的类型，如 float 或 unsigned char

void GlTex2Cuda(int glTex, int width, int height, CUdeviceptr device_frame_ptr, size_t pitch) {
    // Step 1: 注册 OpenGL 纹理为 CUDA 图形资源
    cudaGraphicsResource_t cudaResource;
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    if (err != cudaSuccess) {
        // 处理错误
        printf("texid to cuda err0!");
        return;
    }
    // Step 2: 映射资源
    err = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (err != cudaSuccess) {
        // 处理错误
        printf("texid to cuda err1!");
        return;
    }
    // Step 3: 获取 CUDA 可用的数组
    cudaArray_t texture_ptr;
    err = cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaResource, 0, 0);
    if (err != cudaSuccess) {
        // 处理错误
        printf("texid to cuda err2!");
        return;
    }

    // Step 4: 将 CUDA 纹理数据拷贝到另一个 CUDA 设备内存
    err = cudaMemcpy2DFromArray(device_frame_ptr, pitch, texture_ptr, 0, 0, width * sizeof(pixelType), height, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        // 处理错误
        // printf("texid to cuda err3!");
        std::cerr << "cudaMemcpy2DFromArray failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // Step 5: 解锁资源
    err = cudaGraphicsUnmapResources(1, &cudaResource, 0);
    if (err != cudaSuccess) {
        // 处理错误
        printf("texid to cuda err4!");
        return;
    }
    // 可选: 注销资源
    err = cudaGraphicsUnregisterResource(cudaResource);
    if (err != cudaSuccess) {
        // 处理错误
        printf("texid to cuda err5!");
        return;
    }
    // printf("texid to cuda success!");
}


static uint32_t find_start_code(uint8_t *buf, uint32_t zeros_in_startcode)
{
    uint32_t info;
    uint32_t i;

    info = 1;
    if ((info = (buf[zeros_in_startcode] != 1) ? 0 : 1) == 0)
        return 0;

    for (i = 0; i < zeros_in_startcode; i++)
        if (buf[i] != 0) {
            info = 0;
            break;
        };

    return info;
}
static uint8_t *get_nal(uint32_t *len, uint8_t **offset, uint8_t *start, uint32_t total, uint8_t *prefix_len)
{
    uint32_t info;
    uint8_t *q;
    uint8_t *p = *offset;
    uint8_t prefix_len_z = 0;
    *len = 0;
    *prefix_len = 0;
    while (1) {

        if (((p - start) + 3) >= total)
            return NULL;

        info = find_start_code(p, 2);
        if (info == 1) {
            prefix_len_z = 2;
            *prefix_len = prefix_len_z;
            break;
        }

        if (((p - start) + 4) >= total)
            return NULL;

        info = find_start_code(p, 3);
        if (info == 1) {
            prefix_len_z = 3;
            *prefix_len = prefix_len_z;
            break;
        }
        p++;
    }
    q = p;
    p = q + prefix_len_z + 1;
    prefix_len_z = 0;
    while (1) {
        if (((p - start) + 3) >= total) {
            *len = (start + total - q);
            *offset = start + total;
            return q;
        }

        info = find_start_code(p, 2);
        if (info == 1) {
            prefix_len_z = 2;
            break;
        }

        if (((p - start) + 4) >= total) {
            *len = (start + total - q);
            *offset = start + total;
            return q;
        }

        info = find_start_code(p, 3);
        if (info == 1) {
            prefix_len_z = 3;
            break;
        }

        p++;
    }

    *len = (p - q);
    *offset = p;
    return q;
}
static void CreateCudaContext(CUcontext *cuContext, int iGpu, unsigned int flags)
{
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    ck(cuCtxCreate(cuContext, flags, cuDevice));
    return;
}
// 查询显卡是否支持H264编码
static bool SupportHardEnc(int iGpu)
{
    return false;
}
NvCodecRender::NvCodecRender(const char *input, const char *output, int gpu_idx, bool use_nvenc)
{
    cudaSetDevice(gpu_idx);
    in_file_ = input;
    out_file_ = output;
    demuxer_ = new FFmpegDemuxer(input);
    width_ = demuxer_->GetWidth();
    height_ = demuxer_->GetHeight();
    printf("video w:%d , h:%d" , width_,height_);

    static std::once_flag flag;
    gpu_idx_ = gpu_idx;

    std::call_once(flag, [this] {
        CreateCudaContext(&cuContext, this->gpu_idx_, 0);
    });
    dec_ = new NvDecoder(cuContext, true, FFmpeg2NvCodecId(demuxer_->GetVideoCodec()), true);
    //use_nvenc_ = SupportHardEnc(gpu_idx);
    use_nvenc_ = use_nvenc;

    EncInit();
    InitRender();
    InitRenderTexture(glContext.GetWindowWidth(),glContext.GetWindowHiehgt());
    BindTexture(texId0);
    mp4_ = new MP4Writer(output);
}
NvCodecRender::~NvCodecRender()
{
    if (demuxer_) {
        delete demuxer_;
        demuxer_ = NULL;
    }
    if (dec_) {
        delete dec_;
        dec_ = NULL;
    }
    EncDestory();
    if (mp4_) {
        delete mp4_;
        mp4_ = NULL;
    }
    if (v_packet_) {
        free(v_packet_);
        v_packet_ = NULL;
    }
    printf("~ReidRender\n");
}
int NvCodecRender::EncInit()
{
    demuxer_->GetParam(v_fps_, v_bitrate_); // 编码的时候使用原始视频的帧率和码流，尽可能的保留原始画质
    if (v_fps_ <= 0) {
        v_fps_ = 25;
    }
    if (v_bitrate_ <= 0) {
        v_bitrate_ = 4000000;
    }
    out_fps_ = v_fps_; // 输出帧率

    std::string param1 = "-codec h264 -preset p4 -profile baseline -tuninginfo ultralowlatency -bf 0 "; // 编码参数，根据需求自行修改
    std::string param2 = "-fps " + std::to_string(v_fps_) + " -gop " + std::to_string(2 * v_fps_) + " -bitrate " + std::to_string(v_bitrate_);
    std::string sz_param = param1 + param2;
    printf("sz_param:%s\n", sz_param.c_str());
    int rgba_frame_pitch = width_ * 4;
    int rgba_frame_size = rgba_frame_pitch * height_;
    ck(cuMemAlloc(&ptr_image_enc_, rgba_frame_size));
    eformat_ = NV_ENC_BUFFER_FORMAT_ABGR; //rgb
    init_param_ = NvEncoderInitParam(sz_param.c_str());
    enc_ = new NvEncoderCuda(cuContext, width_, height_, eformat_);
    NV_ENC_INITIALIZE_PARAMS initialize_params = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encode_config = {NV_ENC_CONFIG_VER};
    initialize_params.encodeConfig = &encode_config;
    enc_->CreateDefaultEncoderParams(&initialize_params, init_param_.GetEncodeGUID(), init_param_.GetPresetGUID(), init_param_.GetTuningInfo());
    init_param_.SetInitParams(&initialize_params, eformat_);
    enc_->CreateEncoder(&initialize_params);

    return 0;
}
int NvCodecRender::EncDestory()
{
    if (enc_) {
        enc_->DestroyEncoder();
        delete enc_;
        enc_ = NULL;
    }

    ck(cuMemFree(ptr_image_enc_));
    ptr_image_enc_ = 0;

    return 0;
}
int NvCodecRender::Write2File(uint8_t *data, int len)
{
    uint8_t *p_video = NULL;
    uint32_t nal_len;
    uint8_t *buf_sffset = data;
    uint8_t prefix_len = 0;
    uint8_t *video_data = data;
    uint32_t video_len = len;
    p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    while (p_video != NULL) {
        prefix_len = prefix_len + 1;
        uint8_t nal_type = p_video[prefix_len] & 0x1f;
        if (nal_type == 7) {
            memcpy(sps_buffer_, p_video + prefix_len, nal_len - prefix_len);
            sps_len_ = nal_len - prefix_len;
        } else if (nal_type == 8) {
            memcpy(pps_buffer_, p_video + prefix_len, nal_len - prefix_len);
            pps_len_ = nal_len - prefix_len;
            if (video_track_ == -1) {
                unsigned char buffer[1024];
                int len = h264_video_record_config(buffer, sps_buffer_, sps_len_, pps_buffer_, pps_len_);
                video_track_ = mp4_->AddVideo(width_, height_, buffer, len, H264);
            }
        } else if (nal_type == 5 || nal_type == 1) {
            bool is_key = nal_type == 5 ? true : false;
            v_pts_ += 1000 / out_fps_;
            uint32_t packet_len = nal_len - prefix_len + 4;
            if (v_packet_ == NULL || packet_len > packet_len_) {
                packet_len_ = packet_len > packet_len_ ? packet_len : packet_len_;
                v_packet_ = (unsigned char *)realloc(v_packet_, packet_len_);
            }
            v_packet_[0] = (nal_len - prefix_len) >> 24;
            v_packet_[1] = (nal_len - prefix_len) >> 16;
            v_packet_[2] = (nal_len - prefix_len) >> 8;
            v_packet_[3] = (nal_len - prefix_len) & 0xff;
            memcpy(v_packet_ + 4, p_video + prefix_len, nal_len - prefix_len);
            mp4_->WriteMedia(v_packet_, packet_len, video_track_, v_pts_, v_pts_, is_key);
        }
        p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    }
    return 0;
}
int NvCodecRender::EncFrame(void *ptr, int size)
{
    std::vector<std::vector<uint8_t>> vPacket;
    if (ptr != NULL) {
        const NvEncInputFrame *encoder_input_frame = enc_->GetNextInputFrame();
        //printf("*** encframe *** %d , %d" ,  enc_->GetEncodeWidth(),enc_->GetEncodeHeight());
        NvEncoderCuda::CopyToDeviceFrame(cuContext, ptr, 0, (CUdeviceptr)encoder_input_frame->inputPtr,
                                            (int)encoder_input_frame->pitch,
                                            enc_->GetEncodeWidth(),
                                            enc_->GetEncodeHeight(),
                                            CU_MEMORYTYPE_DEVICE, // CU_MEMORYTYPE_HOST,CU_MEMORYTYPE_DEVICE
                                            encoder_input_frame->bufferFormat,
                                            encoder_input_frame->chromaOffsets,
                                            encoder_input_frame->numChromaPlanes);
        enc_->EncodeFrame(vPacket);
        // printf("enc frame pitch 111 %d\n" , (int)encoder_input_frame->pitch);
    } else {
        enc_->EndEncode(vPacket);
    }
    for (std::vector<uint8_t> &packet : vPacket) {
        // write to file
        Write2File(packet.data(), packet.size());
    } 
    return 0;
}
int NvCodecRender::Draw(unsigned char *rgba_frame, int w, int h)
{
    // cv::Mat image(h, w, CV_8UC4, rgba_frame);

    // cv::Rect rect(200, 200, 200, 200);
    // cv::Scalar rect_color(0, 255, 0);
    // cv::rectangle(image, rect, rect_color, 2);

    // cv::Scalar text_color(0, 255, 0);
    // cv::putText(image, "BreakingY:kxsun@163.com", cv::Point(200, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
    DoRender();
    // BindTexture(texId0);
    // glReadPixels(0, 0, width_, height_, GL_RGBA,GL_UNSIGNED_BYTE,rgba_frame);
    return 0;
}
int NvCodecRender::Render()
{
    CUdeviceptr dp_rgba_frame = 0;
    std::unique_ptr<uint8_t[]> p_rgba_frame;
    // rgba
    int rgba_frame_pitch = width_ * 4;
    int rgba_frame_size = rgba_frame_pitch * height_;
    ck(cuMemAlloc(&dp_rgba_frame, rgba_frame_size));
    p_rgba_frame.reset(new uint8_t[rgba_frame_size]);

    uint8_t *p_video = NULL;
    int n_video_bytes = 0;
    
    do {
        int64_t pts;
        demuxer_->Demux(&p_video, &n_video_bytes, &pts);
        // uint8_t *p_frame;
        int n_frame_returned = 1;
        // n_frame_returned = dec_->Decode(n_video_bytes > 0 ? p_video : NULL, n_video_bytes, CUVID_PKT_ENDOFPICTURE | CUVID_PKT_TIMESTAMP, pts); // CUVID_PKT_ENDOFPICTURE解码器立即输出，没有缓存，没有解码缓存时延;CUVID_PKT_TIMESTAMP返回原始时间戳
        // int i_matrix = dec_->GetVideoFormatInfo().video_signal_description.matrix_coefficients;
        printf("*** n_video_bytes *** %d" , n_video_bytes);
        for (int i = 0; i < n_frame_returned; i++) {
            int64_t timestamp;
            // p_frame = dec_->GetFrame(&timestamp);
            // printf("Dec output timestamp:%ld\n", timestamp);
            total_frames_++;
            // Nv12ToColor32<RGBA32>(p_frame, width_, (uint8_t *)dp_rgba_frame, rgba_frame_pitch, width_, height_, i_matrix);

            //将结果从 GPU 内存传回 CPU 内存
            //ck(cuMemcpyDtoH(p_rgba_frame.get(), dp_rgba_frame, rgba_frame_size));
            //Draw(p_rgba_frame.get(), width_, height_);
            DoRender();
            // glFinish();

            CUdeviceptr device_frame_ptr;
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_frame_ptr),rgba_frame_size);
            if (err != cudaSuccess) {
                printf("texid to cuda err2222!");
                return ;
            }

            // cudaMalloc((void**)&device_frame_ptr,rgba_frame_size);
            const NvEncInputFrame *encoder_input_frame = enc_->GetNextInputFrame();
            GlTex2Cuda(texId0, width_, height_, device_frame_ptr, (int)encoder_input_frame->pitch);
            EncFrame(device_frame_ptr, rgba_frame_size);
       
        }
    } while (n_video_bytes);
    EncFrame(NULL, 0);
    // clear
    ck(cuMemFree(dp_rgba_frame));
    dp_rgba_frame = 0;
    p_rgba_frame.reset(nullptr);
    return 0;
}