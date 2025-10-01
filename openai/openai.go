package openai

import (
	"fmt"
	"sync"
)

var once sync.Once

func main() {
	fmt.Println("Hello, World!")
}
