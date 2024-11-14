#include "ContextEGL.h"


#include <iostream>
#include <string>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>


#define DEBUG(x) std::cout <<  x << "\n";
#define ERROR(x) std::cout <<  x << "\n";

ContextEGL::ContextEGL():m_Driver(GLES_3_0),
                                        m_EglDisplay(EGL_NO_DISPLAY),
		                                m_EglContext(EGL_NO_CONTEXT),
		                                m_EglSurface(EGL_NO_SURFACE)
{

}

ContextEGL::~ContextEGL(){
    CleanUp();
}

void ContextEGL::MakeCurrent(){
    eglMakeCurrent(m_EglDisplay, m_EglSurface, m_EglSurface, m_EglContext);
}

void ContextEGL::ReleaseCurrent(){
    eglMakeCurrent(m_EglDisplay,EGL_NO_SURFACE,EGL_NO_SURFACE,m_EglContext);
}

void ContextEGL::CleanUp() {
	if (m_EglDisplay != EGL_NO_DISPLAY && m_EglSurface != EGL_NO_SURFACE) {
		eglDestroySurface(m_EglDisplay, m_EglSurface);
		m_EglSurface = EGL_NO_SURFACE;
	}

	if (m_EglDisplay != EGL_NO_DISPLAY && m_EglContext != EGL_NO_CONTEXT) {
		eglDestroyContext(m_EglDisplay, m_EglContext);
		m_EglContext = EGL_NO_CONTEXT;
	}

	if (m_EglDisplay != EGL_NO_DISPLAY) {
		eglTerminate(m_EglDisplay);
		m_EglDisplay = EGL_NO_DISPLAY;
	}
};

int ContextEGL::GetWindowWidth(){
    return m_Width;
}

int ContextEGL::GetWindowHiehgt(){
    return m_Height;
}

Error ContextEGL::Initialize(){

    DEBUG("ContextEGL Initialize!");

    EGLint configAttribList[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, //eglCreatePbufferSurface() need this
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,     //开启alpha
		EGL_DEPTH_SIZE, 24,
		EGL_STENCIL_SIZE, 8,
		EGL_SAMPLE_BUFFERS, 1, //开启多重采样
        EGL_SAMPLES, 4,        //多重采样数量
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_NONE
	};

	EGLint numConfigs = 0;
	EGLint majorVersion;
	EGLint minorVersion;

	EGLDisplay display = EGL_NO_DISPLAY;
	EGLContext context = EGL_NO_CONTEXT;
	EGLSurface surface = EGL_NO_SURFACE;
	EGLConfig config = nullptr;
	EGLint contextAttribs[3];
	if (m_Driver == GLES_2_0) {
		contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
		contextAttribs[1] = 2;
		contextAttribs[2] = EGL_NONE;
	} else {
		contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
		contextAttribs[1] = 3;
		contextAttribs[2] = EGL_NONE;
	}

    const int windowWidth = 1280;
    const int windowHeight = 720;

    EGLint pbufferAttribList[] = {
        EGL_WIDTH, windowWidth,
        EGL_HEIGHT, windowHeight,
        EGL_NONE,
    };

    {
        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (!eglGetPlatformDisplayEXT) {
            ERROR("Failed to get function eglGetPlatformDisplayEXT");
            return FAILED;
        }

        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, EGL_DEFAULT_DISPLAY, 0);
        if (display == EGL_NO_DISPLAY) {
            ERROR("Failed to get default EGL display");
            return FAILED;
        }

        if (eglInitialize(display, &majorVersion, &minorVersion) == EGL_FALSE) {
			ERROR("Failed to initialize EGL");
            return FAILED;
		}
        printf("****** EGL version: %d.%d ****** \n", majorVersion, minorVersion);

        if (eglGetConfigs(display, NULL, 0, &numConfigs) == EGL_FALSE) {
			//num is 65
            ERROR("Failed to get EGLConfig count");
            return FAILED;
		}

        if (eglChooseConfig(display, configAttribList, &config, 1, &numConfigs) == EGL_FALSE) {
            //num is 1
			ERROR("Failed to choose first EGLConfig count");
            return FAILED;
		}

		surface = eglCreatePbufferSurface(display, config, pbufferAttribList);
		if (surface == EGL_NO_SURFACE) {
			ERROR("Failed to create EGL fullscreen surface");
            return FAILED;
		}

		context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
		if (context == EGL_NO_CONTEXT) {
			ERROR("Failed to create EGL context");
            return FAILED;
		}

		if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
			ERROR("Failed to make fullscreen EGLSurface current");
            return FAILED;
		}

        m_EglDisplay = display;
	    m_EglSurface = surface;
	    m_EglContext = context;
        DEBUG("Initialize Success!");

        EGLBoolean result = eglQuerySurface(display, surface, EGL_WIDTH, &m_Width);
	    result &= eglQuerySurface(display, surface, EGL_HEIGHT, &m_Height);
        // if(result != EGL_FALSE){
            printf("Display width : %d , height %d \n",static_cast<int>(m_Width), static_cast<int>(m_Height));
        //}

        return OK;
    }
}