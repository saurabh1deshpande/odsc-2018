from pong_nn import PongAgent

def main():
    agent = PongAgent()
    agent.load_model('./model/final_model.pth')
    agent.play(3)
    agent.close_env()

if __name__ == "__main__":
    main()