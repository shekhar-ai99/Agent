{
  type: "AdaptiveCard",
  body: [
    {
        type: "TextBlock",
        text: "Order Details : "&First(Topic.PageData).SalesOrderNumber,
        weight: "bolder",
        size: "medium"
      },
    {
      type: "Container",
      spacing: "Large",
      style: "emphasis",
      items: [
        {
          type: "ColumnSet",
          columns: [
            {
              type: "Column",
              width: "stretch",
              items: [
                {
                  type: "TextBlock",
                  weight: "Bolder",
                  text: "Material Description",
                  wrap: true
                  
                }
              ]
              
            },
            {
              type: "Column",
              spacing: "Large",
              width: "auto",
              items: [
                {
                  type: "TextBlock",
                  weight: "Bolder",
                  text: "Status",
                  wrap: true,
                  horizontalAlignment: "Center",
                  spacing: "Small"
                }
              ]
            },
            {
              type: "Column",
              items: [
                {
                  type: "TextBlock",
                  weight: "Bolder",
                  text: "",
                  wrap: true
                }
              ],
              width: "auto"
            },
             
          ]
        }
      ],
      bleed: true
    },
    {
      
      type: "Container",
      items: ForAll(Topic.PageData,
       {
        type:"Container",
        items: [
          {
          type: "ColumnSet",
          columns: [
            {
              type: "Column",
              items: [
                {
                  type: "TextBlock",
                  text: MaterialDescription,
                  wrap: true
                }
              ],
              width: "stretch",
              horizontalAlignment: "Left"
            },
            {
              type: "Column",
              spacing: "Medium",
              items: [
                {
                  type: "TextBlock",
                  text: Status,
                  wrap: true,
                  horizontalAlignment: "Center"
                }
              ],
              width: "stretch"
            },
           
            {
              type: "Column",
              spacing: "Small",
              selectAction: {
                type: "Action.ToggleVisibility",
                title: "expand",
                targetElements: ["cardContent"&ItemNumber,"chevronDown"&ItemNumber,"chevronUp"&ItemNumber
                ]
              },
              verticalContentAlignment: "Center",
              items: [
                {
                  type: "Image",
                  id: "chevronDown"&ItemNumber,
                  url: "https://adaptivecards.io/content/down.png",
                  width: "20px",
                  altText: "collapsed",
                  backgroundColor: "a"
                },
                {
                  type: "Image",
                  id: "chevronUp"&ItemNumber,
                  url: "https://adaptivecards.io/content/up.png",
                  width: "20px",
                  altText: "expanded",
                  isVisible: false
                }
              ]
              
              
            }
          ],
          bleed: true,
          separator: true,
          spacing: "Small",
          horizontalAlignment: "Left",
          style: "default"
        },
        {
          
          type: "Container",
          id: "cardContent"&ItemNumber,
          isVisible: false,
          items: [
            {
              type: "Container",
              items: [
                {
                  type: "Table",
                  columns: [
                    {
                      width: "1"
                    },
                    {
                      width: "1"
                    }
                  ],
                  rows: [
                                     
                    {
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "SO Date",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: ThisRecord.CreatedDate,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    },
                      {

                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Quantity",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: OrderQuantity,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    },
					    {
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Net value",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: DocumentCurrency&" "&NetValue,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    }, 
                     {
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Material Number",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: MaterialNumber,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    },
                     
                      {
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Status",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: Status,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    },
{
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Expected Delivery Date",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: ThisRecord.Eta,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    },
					{
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Tracking Link",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: ThisRecord.TrackingInfo,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    }, 
					{
                      type: "TableRow",
                      cells: [
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: "Tracking ID",
                              weight: "Bolder",
                              wrap: true
                            }
                          ]
                        },
                        {
                          type: "TableCell",
                          items: [
                            {
                              type: "TextBlock",
                              text: ThisRecord.TrackingID,
                              wrap: true
                            }
                          ]
                        }
                      ]
                    }
					
                  
                  
                                  ]
                }
              ]
            }
          ]
        }
      ]
       }
      ),
      separator: true
    },

{
  type: "Container",
  items: [
    {
      type: "Container",
      isVisible: Topic.ShowPagination,
      items: [
        {
          type: "TextBlock",
          text: "Page " &Topic.CurrentPage &" of " &Topic.NumPages,
          weight: "bolder",
        size: "medium",horizontalAlignment: "Right"
        }
        
      ]
    }]}
  ],
  '$schema': "http://adaptivecards.io/schemas/adaptive-card.json",
  version: "1.5",
  fallbackText: "This card requires Adaptive Cards v1.5 support to be rendered properly."
}






























































































































































