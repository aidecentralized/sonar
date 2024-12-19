# Sonar Web Fall 2024 Wrap Up Notes
[Notion Semester Wrap Up](https://www.notion.so/Semester-Wrap-Up-Fall-2024-006f7aa1921a4954bee9717fcd8a8955?pvs=4)

## Past
- **Sonar Web Integration with SONAR repo**: [GitHub](https://github.com/aidecentralized/sonar) at `sonar_web_integration`
  - Use the same virtual environment as used to run experiments
  - In server terminal: `python algos/rtc_server.py`
  - In each client terminal: `python main.py`
  - Make sure the configs are set correctly
- **Javascript Client training repo**: [GitHub](https://github.com/aopatric/decai-fe/tree/js-pytorch-demo)
  - Training happens, loss goes down
  - Needs to be looked at closer

## Present
- **Next Meeting**: Jan 8th
- **Goals**:
  - Demo-ready by mid January
  - Wrap up with experiments by end of IAP
  - Maybe target either Euro ML Sys Paper (mid Jan or Feb?) or ICML Workshop (check back in Jan)

## Notes about Sonar Web Integration
* Right now the error seems to be that things are not connecting to each other correctly? But I think we have gotten things to train correctly before
* This version as of 12/17 is using `rtc4.py`
* The sys_config contains a session ID that all clients try to join with
* Changed supernode to be at node id 0, and node id 0 has no neighbors by default. All other nodes has 1 neighbor that is its rank + 1. So we are using num_ready = max_clients + 1 to check if the network is ready. Because of this, besides the server, you also have to open up one extra terminal to start an extra client for supernode 0
* I also commented out some stuff about checking for communities that was throwing errors, (`self.communities`), but might have to add that back
* topology hasn't been implemented yet, right now it's just a default
* javascript client (from the other repo) needs to be implemented
* Things do log into a folder
* main files are: `rtc_server.py, fl_static.py, rtc4.py`