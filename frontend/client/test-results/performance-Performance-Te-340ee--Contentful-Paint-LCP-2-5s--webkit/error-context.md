# Page snapshot

```yaml
- generic [ref=e3]:
  - generic [ref=e4]:
    - generic [ref=e5]:
      - heading "Machine Health Dashboard" [level=3] [ref=e6]
      - generic [ref=e7]:
        - generic [ref=e8]:
          - img [ref=e9]
          - generic [ref=e11]: Online
        - generic [ref=e12]:
          - img [ref=e13]
          - generic [ref=e15]: Disconnected
    - paragraph [ref=e16]: Real-time monitoring and predictive maintenance for industrial equipment
  - generic [ref=e20]:
    - generic [ref=e21]: Select Machine
    - generic [ref=e22]:
      - img [ref=e24]
      - combobox "Select Machine" [ref=e26]
      - button "Open" [ref=e28] [cursor=pointer]:
        - img [ref=e29]
      - group:
        - generic: Select Machine
  - generic [ref=e31]:
    - heading "Select a Machine to Begin Monitoring" [level=5] [ref=e32]
    - paragraph [ref=e33]: Choose a machine from the dropdown above to view real-time sensor data, run predictions, and monitor health status
  - alert [ref=e34]:
    - img [ref=e36]
    - generic [ref=e38]: Failed to load machines from server. Using fallback data.
    - button "Close" [ref=e40] [cursor=pointer]:
      - img [ref=e41]
```