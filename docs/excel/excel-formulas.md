### In excel diff two timestamps with date to get difference in hours
To calculate the difference in hours between two timestamps (including dates) in Excel, subtract the start timestamp from the end timestamp and multiply by 24. This works for spans across multiple days and returns a decimal value like 30.5 for 30 hours and 30 minutes. [support.microsoft](https://support.microsoft.com/en-us/office/calculate-the-difference-between-two-times-in-excel-e1c78778-749b-49a3-b13e-737715505ff6)

## Basic Formula
Assume start timestamp in A2 (e.g., "4/27/2026 9:00 AM") and end in B2 (e.g., "4/28/2026 3:30 PM").

Use `=(B2 - A2) * 24` in C2.

Format C2 as Number for decimal hours (e.g., 30.5). [zapier](https://zapier.com/blog/calculate-time-difference-in-excel/)

## Example Results
| Start Timestamp | End Timestamp | Formula Result (Hours) |
|-----------------|---------------|------------------------|
| 4/27/26 10:00  | 4/27/26 14:00 | 4                      |
| 4/27/26 22:00  | 4/28/26 06:00 | 8                      |
| 4/27/26 09:00  | 4/29/26 15:30 | 54.5                   |  [support.microsoft](https://support.microsoft.com/en-us/office/calculate-the-difference-between-two-times-in-excel-e1c78778-749b-49a3-b13e-737715505ff6)

## Tips
Ensure cells A2:B2 are formatted as Date/Time (right-click > Format Cells > Custom > "m/d/yyyy h:mm AM/PM").

For whole hours only, wrap in INT: `=INT((B2 - A2) * 24)`. Negative results? Use ABS or swap cells. [ablebits](https://www.ablebits.com/office-addins-blog/calculate-time-excel/)
