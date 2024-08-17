# SONAR: Self-Organizing Network of Aggregated Representations
**Project by MIT Media Lab**

A collaborative learning project where users self-organize to improve their ML models by sharing representations of their data or model. 

To get started, please refer to the [Get Started](./getting-started/getting-started.md) page.

## Main
The application currently uses MPI and GRPC (experimental) to enable communication between different nodes in the network. The goal of the framework to organize everything in a modular manner. That way a researcher or engineer can easily swap out different components of the framework to test their hypothesis or a new algorithm.

| Topology                   | Train                  |                     |                     | Test                   |                     |                     |
|----------------------------|------------------------|---------------------|---------------------|------------------------|---------------------|---------------------|
|                            | 1                      | 2                   | 3                   | 1                      | 2                   | 3                   |
| Isolated                   | 208.0<sub>(0.3)</sub>  | 208.0<sub>(0.3)</sub>| 208.0<sub>(0.0)</sub>| 44.5<sub>(6.9)</sub>  | 44.5<sub>(6.9)</sub>| 44.5<sub>(6.9)</sub>|
| Central*                   | 208.5<sub>(0.1)</sub>  | 208.5<sub>(0.0)</sub>| 208.5<sub>(0.0)</sub>| 33.97<sub>(14.2)</sub>| 33.97<sub>(14.2)</sub>| 33.97<sub>(14.2)</sub>|
| Random                     | 205.4<sub>(1.0)</sub>  | 205.5<sub>(0.9)</sub>| 205.9<sub>(0.8)</sub>| 54.9<sub>(5.3)</sub>  | 56.0<sub>(5.8)</sub>| 56.2<sub>(5.6)</sub>|
| Ring                       | 198.8<sub>(3.1)</sub>  | 198.7<sub>(3.3)</sub>| 198.7<sub>(3.4)</sub>| 47.8<sub>(7.3)</sub>  | 46.9<sub>(6.9)</sub>| 47.6<sub>(7.1)</sub>|
| Grid                       | 202.6<sub>(1.5)</sub>  | 203.9<sub>(1.4)</sub>| 204.5<sub>(1.3)</sub>| 49.3<sub>(6.0)</sub>  | 48.8<sub>(6.0)</sub>| 48.1<sub>(6.1)</sub>|
| Torus                      | 202.0<sub>(1.2)</sub>  | 203.2<sub>(1.2)</sub>| 204.0<sub>(1.3)</sub>| 50.2<sub>(6.0)</sub>  | 50.7<sub>(6.6)</sub>| 50.3<sub>(6.2)</sub>|
| Similarity based (top-k)   | 206.4<sub>(1.6)</sub>  | 197.6<sub>(7.3)</sub>| 200.4<sub>(4.4)</sub>| 47.3<sub>(8.3)</sub>  | 48.4<sub>(8.5)</sub>| 52.8<sub>(7.2)</sub>|
| Swarm                      | 183.2<sub>(3.5)</sub>  | 183.1<sub>(3.6)</sub>| 183.2<sub>(3.6)</sub>| 52.2<sub>(8.7)</sub>  | 52.3<sub>(8.7)</sub>| 52.4<sub>(8.6)</sub>|
| L2C                        | 167.0<sub>(25.4)</sub> | 158.8<sub>(30.6)</sub>| 152.8<sub>(35.2)</sub>| 37.6<sub>(7.4)</sub>  | 36.6<sub>(7.4)</sub>| 35.8<sub>(7.7)</sub>|

**Table 1:** Performance overview (AUC) of various topologies with different number of collaborators.

**Table 2** Area Under Curve of Test Accuracy Varying Number of Users
| Num Users |      DomainNet                            |     Camelyon17                           |      Digit-Five                         |
|-----------|----------------------|--------------------|---------------------|--------------------|----------------------|------------------|
|           | Within Domain        | Random             | Within Domain       | Random             | Within Domain        | Random           |
|-----------|----------------------|--------------------|---------------------|--------------------|----------------------|------------------|
|     12    | 56.6267              | 50.1772            | 178.3622            | 145.6398           | 57.1168              | 68.9724          |
|     18    | 59.5647              | 54.2480            | 179.5941            | 145.9916           | 66.8201              | 69.8341          |
|     24    | 61.8006              | 54.3855            | 178.5976            | 149.2037           | 71.6536              | 72.5333          |
|     30    | 66.5896              | 58.4835            | 179.1761            | 153.0658           | 74.4239              | 72.6996          |
|     39    | 68.3743              | 59.6090            | 179.1404            | 149.8618           | 163.8116             | 163.9892         |
|     45    | 68.124               | 59.7852            | 180.0231            | 147.4649           | 77.0248              | 73.0634          |


**Table 3*** Area Under Curve of Test Accuracy Varying Number of Domains

|   AUC DomainNet (48 users, 200 rounds)      |
|---------------------------------------------|
| Num Domains | Within Domain | Random        |
|-------------|---------------|---------------|
|   2         | 67.7514       | 58.7947       |
|   4         | 61.5723       | 50.2906       |
|   6         | 69.4671       | 47.7867       |

|   AUC Camelyon17 (30 users, 200 rounds)           |
|---------------------------------------------------|
| Num Domains | Within Domain       | Random        |
|-------------|---------------------|---------------|
|      2      | 179.7901            | 172.9167      |
|      3      | 179.1761            | 153.0658      |
|      5      | 176.5059            | 139.4547      |

|    AUC Digit-Five (30 users, 200 rounds)          |
|---------------------------------------------------|
| Num Domains | Within Domain       | Random        |
|-------------|---------------------|---------------|
|      2      | 71.8536             | 65.6555       |
|      3      | 74.4239             | 72.6996       |
|      5      | 77.3709             | 76.3041       |

**Table 4** Test Accuracy and Standard Deviation Over Rounds

| Rounds |          DomainNet (39 users, 3 domains)           |               |               |                    |
|--------|----------------------------------------------------|---------------|---------------|---------------|
|        |                  Within Domain    |                |                Random             |               |
|--------|---------------------------|------------------------|---------------------------|------------------------|
|        | Mean                      | Std                    | Mean                      | Std                    |
| 100    | 0.3619                    | 0.0635                 | 0.3212                    | 0.0625                 |
| 200    | 0.4220                    | 0.0563                 | 0.3733                    | 0.0791                 |
| 300    | 0.4362                    | 0.0498                 | 0.4203                    | 0.0537                 |
| 400    | 0.4353                    | 0.0687                 | 0.4355                    | 0.0585                 |
| 500    | 0.4726                    | 0.0502                 | 0.4499                    | 0.0496                 |



| Rounds |          Camelyon17 (39 users, 3 domains)          |               |               |               |
|--------|-----------------------------------------------------|---------------|---------------|---------------|
|        |                  Within Domain                      |               |                Random             |               |
|--------|---------------------------|------------------------|---------------------------|------------------------|
|        | Mean                      | Std                    | Mean                      | Std                    |
| 40     | 0.9086                    | 0.0255                 | 0.7281                    | 0.1405                 |
| 80     | 0.9169                    | 0.0251                 | 0.7460                    | 0.1196                 |
| 120    | 0.9329                    | 0.0195                 | 0.7293                    | 0.1520                 |
| 160    | 0.9361                    | 0.0239                 | 0.8122                    | 0.1346                 |
| 200    | 0.9353                    | 0.0251                 | 0.7762                    | 0.1516                 |


|        |          Digit Five (39 users, 3 domains)         |                                                   |
|--------|---------------------------------------------------|---------------------------------------------------|
| Rounds |             Within Domain |                       |                      Random                       |
|--------|---------------------------|-----------------------|---------------------------|-----------------------|
|        | Mean                      | Std                   | Mean                      | Std                   |
| 20     | 0.7314                    | 0.1290                | 0.6788                    | 0.0839                |
| 40     | 0.8080                    | 0.0974                | 0.8151                    | 0.0549                |
| 60     | 0.8350                    | 0.0937                | 0.8464                    | 0.0558                |
| 80     | 0.8417                    | 0.0928                | 0.8673                    | 0.0454                |
| 100    | 0.8310                    | 0.1030                | 0.8733                    | 0.0502                |
