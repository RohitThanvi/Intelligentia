# Loan Processing Standard Operating Procedure

## Intake
All loan applications arrive through the CRM (SalesForceX) or the customer
portal. An operations analyst manually reviews each application within 4 business
hours of receipt to confirm all mandatory fields are present.

## Document Collection
Applicants must submit: income proof, identity documents, bank statements
(last 6 months), and address proof. Approximately 18% of applications are
missing at least one required document or contain inconsistent information
across documents, triggering a manual follow-up communication cycle that
averages 1.5 business days per round-trip.

## Verification
Verification staff cross-check submitted documents against the core banking
system and external bureau data. This step is largely manual and involves
reading unstructured PDF/scanned documents.

## Exception Handling
Exceptions (mismatched income, suspected fraud flags, incomplete KYC) are
routed to a senior underwriter queue. This queue currently has the longest
average dwell time in the process, contributing significantly to the overall
2-5 business day processing window.

## Systems in Use
- CRM: SalesForceX
- Loan Origination Platform: internally built, integrates with core banking
- Document Management System: SharePoint-based repository
- Workflow Engine: Camunda
- Data Warehouse: Enterprise DWH (nightly batch)

## Current Staffing
120 operations employees are involved across document collection,
verification, communication, exception handling, and application processing.
