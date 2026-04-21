# Notes on OpenAI's Responses-API return objects

## Overview: streaming vs. non-streaming

### Non-streaming

The API returns a single Response object. Response has an output=[...] list containing, eg, `ResponseReasoningItem` and `ResponseOutputMessage`

### Streaming

The API returns an iterator yielding ~100 to >1000 ResponseStreamEvent objects. 

Among the event objects, notable `ResponseOutputItemAddedEvent` / `ResponseOutputItemDoneEvent` have an item= argument which contains non-streaming output items, such as the two just mentioned.

The stream concludes with a `ResponseCompletedEvent` wrapping a Response object, identical to the sort obtained in non-streaming mode.

This means there is a lot of redundancy over the wire, with, e.g., `ResponseTextDoneEvent`, `ResponseContentPartDoneEvent`, `ResponseOutputItemDoneEvent`, and `ResponseCompletedEvent` *all* containing the full text of the model's response.


## Sample of GPT-5.2 streaming response events:

Still need to add a function call (type=.function_call_arguments.delta)

```
ResponseCreatedEvent - type='response.created',
                       response=Response(
                           id='resp_0cab51075dc299f80069c7e8c763cc8190b0a2c1604c548c0c'
                           [all parameters for the request]
                       )
ResponseInProgressEvent - type='response.in_progress',
                          response=Response(
                              id='resp_0cab51075dc299f80069c7e8c763cc8190b0a2c1604c548c0c'
                              [all parameters for the request]
                          )
ResponseOutputItemAddedEvent - type='response.output_item.added', output_index=0,
                               item=ResponseReasoningItem(
                                   id='rs_0cab51075dc299f80069c7e8c7e8dc8190b17063f2145009c8'
                                   summary=[]
                                   encrypted_content='...'
                               )
>>>> I presume first reasoning step is now done, and summarizer has started 
ResponseReasoningSummaryPartAddedEvent - type='response.reasoning_summary_part.added', output_index=0
ResponseReasoningSummaryTextDeltaEvent - type='response.reasoning_summary_text.delta', output_index=0,
                                         delta='**Full Multitoken Title**\n\n[one token of body]'
ResponseReasoningSummaryTextDeltaEvent - delta=[one token], output_index=0
    ... x 50
ResponseReasoningSummaryTextDoneEvent - type='response.reasoning_summary_text.done', output_index=0,
                                        text='**Full Multitoken Title**\n\nFull body text'
ResponseReasoningSummaryPartDoneEvent - type='response.reasoning_summary_part.done', output_index=0,
                                        part=Part(text='**Full Multitoken Title**\n\nFull body text')
ResponseOutputItemDoneEvent - type='response.output_item.done', output_index=0,
                              item=ResponseReasoningItem(
                                  id='rs_0cab51075dc299f80069c7e8c7e8dc8190b17063f2145009c8'
                                  summary=[Summary(text='**Full Multitoken Title**\n\nFull body text')]
                                  encrypted_content='...(LONGER THAN IN ResponseOutputItemAddedEvent)'
                              )
ResponseOutputItemAddedEvent - type='response.output_item.added', output_index=1,
                               item=ResponseOutputMessage(
                                  id='msg_0cab51075dc299f80069c7e8cd56908190818b051112de06a1'
                               )
ResponseContentPartAddedEvent - type='response.content_part.added', output_index=1,
ResponseTextDeltaEvent - type='response.output_text.delta', output_index=1,
                         delta=[one token]
                         obfustation=[padding to prevent side-channel attacks on token length]
    ... x 100
ResponseTextDoneEvent - type='response.output_text.done', output_index=1,
                        text='FULL OUTPUT TEXT'
ResponseContentPartDoneEvent - type='response.content_part.done', output_index=1,
                               part=ResponseOutputText(text='FULL OUTPUT TEXT')
ResponseOutputItemDoneEvent - type='response.output_item.done', output_index=1,
                              item=ResponseOutputMessage(
                                  id='msg_0cab51075dc299f80069c7e8cd56908190818b051112de06a1'
                                  content=[ResponseOutputText(text='FULL OUTPUT TEXT')]
                              )
ResponseCompletedEvent -
                                  
                                  encrypted_content='...(SAME LENGTH, DIFF STRING AS ResponseOutputItemDoneEvent)'

```
