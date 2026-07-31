"""Minimal deterministic DOCX writer for BSIP manuscript drafts."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .enums import SectionType
from .models import ManuscriptDocument, ManuscriptSourcePackage


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def write_docx(path: Path, document: ManuscriptDocument, source: ManuscriptSourcePackage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "[Content_Types].xml": _content_types_xml(),
        "_rels/.rels": _root_rels_xml(),
        "docProps/core.xml": _core_xml(document),
        "docProps/app.xml": _app_xml(),
        "word/document.xml": _document_xml(document, source),
        "word/styles.xml": _styles_xml(),
        "word/settings.xml": _settings_xml(),
        "word/_rels/document.xml.rels": _document_rels_xml(),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _document_xml(document: ManuscriptDocument, source: ManuscriptSourcePackage) -> str:
    body = []
    body.append(_paragraph(document.title, style="Title"))
    body.append(_paragraph("Generated under BSIP Reviewer Constraints", style="Subtitle"))
    body.append(_paragraph(f"Document status: {document.document_status.value}", style="Body"))
    author = document.metadata.get("author")
    if author:
        body.append(_paragraph(f"Author: {author}", style="Body"))
    body.append(_paragraph(f"Created: {document.created_at}", style="Body"))
    body.append(_paragraph("This document is an internal scientific draft. It is not labelled as submission ready.", style="Body"))
    body.append(_page_break())
    body.append(_paragraph("Revision Warning", style="Heading1"))
    warning = (
        f"This draft preserves {len(document.unresolved_flags)} unresolved reviewer flag(s). "
        f"The overall reviewer recommendation is {document.metadata.get('overall_reviewer_recommendation')}. "
        "Reviewer blockers have not been marked as resolved."
    )
    body.append(_one_cell_table(warning))
    for section in document.sections:
        if section.section_type is SectionType.TITLE:
            continue
        body.append(_paragraph(section.title, style="Heading1"))
        for sentence in section.sentences:
            if sentence.metadata.get("placeholder"):
                body.append(_paragraph(sentence.text, style="Quote"))
            else:
                body.append(_paragraph(sentence.text, style="Body"))
    body.append(_page_break())
    body.append(_paragraph("Appendix A: Unresolved Reviewer Findings", style="Heading1"))
    for finding in source.review_findings:
        if finding.get("severity") not in {"CRITICAL", "MAJOR", "MODERATE"} and finding.get("blocking") is not True:
            continue
        body.append(_paragraph(str(finding.get("finding_id")), style="Heading2"))
        body.append(_paragraph(f"Severity: {finding.get('severity')}; blocking: {finding.get('blocking')}", style="Body"))
        body.append(_paragraph(str(finding.get("finding_text") or finding.get("title") or ""), style="Body"))
    body.append(_paragraph("Appendix B: Sentence Traceability Summary", style="Heading1"))
    body.append(_traceability_table(document))
    body.append(_sect_pr())
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>'
        + "".join(body)
        + "</w:body></w:document>"
    )


def _paragraph(text: str, *, style: str = "Body") -> str:
    return (
        "<w:p>"
        f'<w:pPr><w:pStyle w:val="{escape(style)}"/></w:pPr>'
        "<w:r>"
        f"<w:t>{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def _one_cell_table(text: str) -> str:
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="9B1C1C"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="9B1C1C"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="9B1C1C"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="9B1C1C"/>'
        "</w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w=\"9360\"/></w:tblGrid>"
        '<w:tr><w:tc><w:tcPr><w:tcW w:w="9360" w:type="dxa"/>'
        '<w:shd w:fill="F4F6F9"/></w:tcPr>'
        + _paragraph(text, style="Body")
        + "</w:tc></w:tr></w:tbl>"
    )


def _traceability_table(document: ManuscriptDocument) -> str:
    rows = [
        ("Sentence ID", "Section", "Traceability", "Source count"),
        *(
            (
                sentence.sentence_id,
                sentence.section_id,
                sentence.traceability_status.value,
                str(len(sentence.source_ids)),
            )
            for sentence in document.sentences[:30]
        ),
    ]
    cell_widths = (2400, 2400, 2400, 2160)
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in cell_widths)
    body = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/></w:tblPr>']
    body.append(f"<w:tblGrid>{grid}</w:tblGrid>")
    for index, row in enumerate(rows):
        body.append("<w:tr>")
        for value, width in zip(row, cell_widths):
            shade = '<w:shd w:fill="F2F4F7"/>' if index == 0 else ""
            body.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>{_paragraph(value, style="Body")}</w:tc>')
        body.append("</w:tr>")
    body.append("</w:tbl>")
    return "".join(body)


def _sect_pr() -> str:
    return (
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Body">
    <w:name w:val="Body"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="0B2545"/><w:sz w:val="34"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="555555"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Quote">
    <w:name w:val="Quote"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="360"/><w:spacing w:before="80" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:i/><w:color w:val="555555"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/><w:left w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/><w:right w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/></w:tblBorders><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar></w:tblPr>
  </w:style>
</w:styles>"""


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def _root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""


def _settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}"><w:zoom w:percent="100"/></w:settings>"""


def _core_xml(document: ManuscriptDocument) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(document.title)}</dc:title>
  <dc:creator>BSIP Manuscript Engine</dc:creator>
  <cp:lastModifiedBy>BSIP Manuscript Engine</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{escape(document.created_at)}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{escape(document.created_at)}</dcterms:modified>
</cp:coreProperties>"""


def _app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>BSIP Manuscript Engine</Application>
</Properties>"""
