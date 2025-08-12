# pdfparser

### High level Goal when done

This repos will create a webhook that will accept a payload from a supabase instance, loading the pdf links it shares, turning them into images, then passing the images in batches to an OPENAI model to determine if the docuemnt meets specific criteria

### Lower level beginning

Implement repos containing a server that can be used in railway.app

Set up a webhook to accept the payload from supabase

Set `WEBHOOK_AUTH_TOKEN` to require a matching `Authorization` header

Log all data received to console
