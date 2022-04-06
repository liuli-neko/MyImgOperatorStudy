#include <GL/glut.h>

void init(void) {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glOrtho(-5, 5, -5, 5, 5, 15);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);

  return;
}

void display(void) {
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0, 0, 0);
  glutWireTeapot(3);
  glFlush();

  return;
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(300, 300);
  glutCreateWindow("OpenGL 3D View");
  init();
  glutDisplayFunc(display);
  glutMainLoop();

  return 0;
}
// ————————————————
// 版权声明：本文为CSDN博主「zhangliang_571」的原创文章，遵循CC 4.0
// BY-SA版权协议，转载请附上原文出处链接及本声明。
// 原文链接：https://blog.csdn.net/zhangliang_571/article/details/25241911/