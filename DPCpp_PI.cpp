
#include <CL/sycl.hpp>
#include "C:\Program Files (x86)\Intel\oneAPI\dpl\2021.1.1\windows\include\oneapi\dpl\random"
// OJO !!! La línea anterior debería funcionar simplemente como sigue:
//#include <oneapi/dpl/random>

using namespace cl::sycl;

int main(int argc, char* argv[]) {
    int length = 1024;
    //int length = 1 << 12;
    clock_t start, end;

    try {
        // cola de trabajo
        sycl::queue q(sycl::default_selector{});//GPU(si hay) o CPU
        //sycl::queue q(sycl::cpu_selector{});
        //sycl::queue q(sycl::gpu_selector{});

        // usamos Unified Shared Memory, USM
        auto counter = sycl::malloc_shared<int>(length, q);

        // inicializamos
        for (int i = 0; i < length; i++) {
            counter[i] = 0;
        }

        // inicia cronómetro
        start = clock();

        // kernel
        q.parallel_for(sycl::range<1>{length}, [=](sycl::id<1> i) {
            // Create minstd_rand engine
            oneapi::dpl::minstd_rand engine(start, i);//start es seed, i es offset
            // Create float uniform_real_distribution distribution
            oneapi::dpl::uniform_real_distribution<float> distr;

            // Calcula los puntos dentro del círculo
            for (int j = 0; j < length; j++) {
                float x = distr(engine);
                float y = distr(engine);
                if (x * x + y * y <= 1.0) {
                    counter[i]++;
                }
            }

            });
        // esperamos que la GPU termine
        q.wait();

        // finaliza cronómetro
        end = clock();

        // calcula total de puntos en el círculo
        int total_puntos = length * length;
        int en_circulo = 0;
        for (int i = 0; i < length; i++) {
            en_circulo += counter[i];
        }
        float PI = 4.0 * ((float)en_circulo / (float)total_puntos);
        printf("Aproximado con %d iteraciones\n", total_puntos);
        printf("%d puntos dentro del círculo\n", en_circulo);
        printf("PI= %f\n", PI);

        // tiempo total de ejecución de la kernel
        double time_taken = double(end - start);
        printf("Clock ticks: %f\n", time_taken);

        // liberamos USM
        sycl::free(counter, q);

    }
    catch (sycl::exception& e) {
        printf("Problemas !!!: %s\n", e.what());
        return 1;
    }

    return 0;
}
