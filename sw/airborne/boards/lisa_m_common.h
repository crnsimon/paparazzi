#ifndef CONFIG_LISA_M_COMMON_H
#define CONFIG_LISA_M_COMMON_H

/* SPI slave mapping */

#define SPI_SELECT_SLAVE0_PERIPH RCC_APB2ENR_IOPAEN
#define SPI_SELECT_SLAVE0_PORT GPIOA
#define SPI_SELECT_SLAVE0_PIN GPIO15

#define SPI_SELECT_SLAVE1_PERIPH RCC_APB2ENR_IOPAEN
#define SPI_SELECT_SLAVE1_PORT GPIOA
#define SPI_SELECT_SLAVE1_PIN GPIO4

#define SPI_SELECT_SLAVE2_PERIPH RCC_APB2ENR_IOPBEN
#define SPI_SELECT_SLAVE2_PORT GPIOB
#define SPI_SELECT_SLAVE2_PIN GPIO12

#define SPI_SELECT_SLAVE3_PERIPH RCC_APB2ENR_IOPCEN
#define SPI_SELECT_SLAVE3_PORT GPIOC
#define SPI_SELECT_SLAVE3_PIN GPIO13

#define SPI_SELECT_SLAVE4_PERIPH RCC_APB2ENR_IOPCEN
#define SPI_SELECT_SLAVE4_PORT GPIOC
#define SPI_SELECT_SLAVE4_PIN GPIO12

#define SPI_SELECT_SLAVE5_PERIPH RCC_APB2ENR_IOPCEN
#define SPI_SELECT_SLAVE5_PORT GPIOC
#define SPI_SELECT_SLAVE5_PIN GPIO4

/* PPM */

#if PPM_CONFIG == 1
#define USE_PPM_TIM1 1
#define PPM_CHANNEL         TIM_IC3
#define PPM_TIMER_INPUT     TIM_IC_IN_TI3
#define PPM_IRQ             NVIC_TIM1_UP_IRQ
#define PPM_IRQ2            NVIC_TIM1_CC_IRQ
#define PPM_IRQ_FLAGS       (TIM_DIER_CC3IE | TIM_DIER_UIE)
#define PPM_GPIO_PERIPHERAL RCC_APB2ENR_IOPAEN
#define PPM_GPIO_PORT       GPIOA
#define PPM_GPIO_PIN        GPIO10

#elif PPM_CONFIG == 2
#define USE_PPM_TIM2 1
#define PPM_CHANNEL         TIM_IC2
#define PPM_TIMER_INPUT     TIM_IC_IN_TI2
#define PPM_IRQ             NVIC_TIM2_IRQ
#define PPM_IRQ_FLAGS       (TIM_DIER_CC2IE | TIM_DIER_UIE)
#define PPM_GPIO_PERIPHERAL RCC_APB2ENR_IOPAEN
#define PPM_GPIO_PORT       GPIOA
#define PPM_GPIO_PIN        GPIO1

// Move default ADC timer
#if USE_AD_TIM2
#undef USE_AD_TIM2
#endif
#define USE_AD_TIM1 1

#else
#error "Unknown PPM config"

#endif // PPM_CONFIG

#endif

