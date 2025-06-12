-- Customer-Level Purchase Behavior Features
SELECT 
    c.CustomerKey,
    g.City,
    g.StateProvinceName,
    g.EnglishCountryRegionName ,
    COUNT(DISTINCT fis.SalesOrderNumber) AS TotalOrders,
    SUM(fis.SalesAmount) AS TotalSales,
    AVG(fis.SalesAmount) AS AvgOrderValue,
    DATEDIFF(DAY, MAX(fis.OrderDate), GETDATE()) AS DaysSinceLastOrder
FROM 
    FactInternetSales fis
JOIN 
    DimCustomer c ON fis.CustomerKey = c.CustomerKey
JOIN 
    DimGeography g ON c.GeographyKey = g.GeographyKey
GROUP BY 
    c.CustomerKey, g.City, g.StateProvinceName, g.EnglishCountryRegionName;