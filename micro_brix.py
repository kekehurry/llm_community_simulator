import websocket
import rel
from time import sleep
import json
from threading import Thread

class MicroMicroBrix(Thread):

    def __init__(self, 
            config,
            host = "ws://localhost:8080",
            quietly=True, 
            module_function = None,
    ):

        super(MicroMicroBrix, self).__init__()
        self.host = host
        self.config = config    
        self.quietly = quietly

        if(module_function == None):
            raise ValueError("module_function should contain a function that returns DeckGL layers")

        self.module_function = module_function

        if(not self.quietly):
            websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp( self.host,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close)
        
    def on_open(self, ws):
        print("## Opened connection")
        self.send_message({"type": "REGISTER_MODULE","content":{"title":self.config["title"],"version":self.config["version"]}})
        

    def on_message(self, ws, message):
        # print("## Received message:",message)
        try:
            message = json.loads(message)
            message_type = message['type']
            if message_type == 'REGISTER_MODULE':
                print("## Registered, #MODULE_ID: ", message['id'])
            elif message_type == 'CONNECTED':
                session_id = message['id']
                self.send_message({"type": "UPDATE_CONFIG","id":session_id, "content":{"config":self.config}})
            elif(message_type == 'UPDATE_INPUT'):
                session_id = message['id']
                input = message['content']['input']
                self.update_output(input, session_id)
        except Exception as e:
            print(e)
            print("## Error parsing message")
            pass

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        print("## Connection closed")

    def send_message(self, message):
        self.ws.send(json.dumps(message))

    def update_output(self,input, session_id):
        try:
            output = self.module_function(input)
            message = {"type": "UPDATE_OUTPUT","id":session_id, "content":{"output": output}}
            self.send_message(message)
        except Exception as e:
            print(e)
            message = {"type": "UPDATE_OUTPUT","id":session_id, "content":{"output": "Error"}}
            self.send_message(message)

    def listen(self):
        self.ws.run_forever(dispatcher=rel, reconnect=5)  
        rel.signal(2, rel.abort)  # Keyboard Interrupt
        rel.dispatch()


def main():

    def get_output(input):
        if "a" in input and "b" in input:
            return {"c": input["a"] + input["b"]}
        else:
            return {"c": "Error"}

    config = {
        "title": "Test Brix Module",
        "version": "0.0.1",
        "input": ["a","b"],
        "output": ["c"]
    }
    
    connection = MicroMicroBrix(config=config,module_function=get_output)
    connection.listen()


if __name__ == "__main__":
    # execute only if run as a script
    main()
