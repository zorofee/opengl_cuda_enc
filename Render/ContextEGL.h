
#include <EGL/egl.h>

enum Error {
	OK,
	FAILED, 
};

class ContextEGL{
public:
	enum Driver {
		GLES_2_0,
		GLES_3_0,
	};
public:
    Error Initialize();
    void MakeCurrent();
    void ReleaseCurrent();
    int GetWindowWidth();
    int GetWindowHiehgt();
    void CleanUp();

    ContextEGL();
    ~ContextEGL();

private:
    EGLDisplay m_EglDisplay;
	EGLContext m_EglContext;
	EGLSurface m_EglSurface;
    EGLint m_Width;
	EGLint m_Height;

    Driver m_Driver;
};