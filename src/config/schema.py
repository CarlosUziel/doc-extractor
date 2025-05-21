"""Schema definitions for document extraction."""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Represents the bounding box coordinates of a text segment.

    Attributes:
        ymin: The y-coordinate of the top-left corner, normalized (0-1000).
        xmin: The x-coordinate of the top-left corner, normalized (0-1000).
        ymax: The y-coordinate of the bottom-right corner, normalized (0-1000).
        xmax: The x-coordinate of the bottom-right corner, normalized (0-1000).
        page_number: The 0-indexed page number of the extraction.
    """

    ymin: int = Field(
        description=(
            "The y value of the top left corner of the box, normalized between 0 and "
            "1000."
        )
    )
    xmin: int = Field(
        description=(
            "The x value of the top left corner of the box, normalized between 0 and "
            "1000."
        )
    )
    ymax: int = Field(
        description=(
            "The y value of the bottom right corner of the box, normalized between "
            "0 and 1000."
        )
    )
    xmax: int = Field(
        description=(
            "The x value of the bottom right corner of the box, normalized between "
            "0 and 1000."
        )
    )
    page_number: Optional[int] = Field(
        default=None,
        description=(
            "The 0-indexed page number from which this bounding box was extracted."
        ),
    )


class AddressDetails(BaseModel):
    """Detailed schema for a property's address.

    Attributes:
        street_number: The street number.
        street_number_bbox: Bounding box for the street number.
        street_name: The name of the street.
        street_name_bbox: Bounding box for the street name.
        zip_code: The postal or zip code.
        zip_code_bbox: Bounding box for the zip code.
        city: The city name.
        city_bbox: Bounding box for the city name.
    """

    street_number: Optional[str] = Field(default=None, description="Street number.")
    street_number_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box for street number, including page number.",
    )
    street_name: Optional[str] = Field(default=None, description="Street name.")
    street_name_bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for street name, including page number."
    )
    zip_code: Optional[str] = Field(default=None, description="Zip code.")
    zip_code_bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for zip code, including page number."
    )
    city: Optional[str] = Field(default=None, description="City.")
    city_bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for city, including page number."
    )


class BasePropertyDocumentDetails(BaseModel):
    """Base schema containing common fields for property-related documents.

    Attributes:
        realty_address: Detailed address of the property.
    """

    realty_address: Optional[AddressDetails] = Field(
        default=None, description="Full address of the realty."
    )

    epc_before_renovations: Optional[int] = Field(
        default=None,
        description=(
            "Energy Performance Certificate (EPC) score of the property before "
            "renovations. This value is an integer, usually between -100 and 500."
        ),
    )
    epc_before_renovations_bbox: Optional[BoundingBox] = Field(
        default=None,
        description=(
            "Bounding box for Energy Performance Certificate (EPC) before renovations, "
            "including page number."
        ),
    )

    living_area_sqm: Optional[float] = Field(
        default=None, description="Surface living area in square meters."
    )
    living_area_sqm_bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for living area, including page number."
    )

    realty_type: Optional[
        Literal["apartment", "constructionLand", "garage", "house", "warehouse"]
    ] = Field(default=None, description="Type of the realty.")
    realty_type_bbox: Optional[BoundingBox] = Field(
        default=None,
        description=(
            "Bounding box for realty type. Select the one that fits best, translate "
            "to English if necessary, including page number."
        ),
    )

    year_of_built: Optional[int] = Field(
        default=None, description="Year when the property was built or constructed."
    )
    year_of_built_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box for year of built, including page number.",
    )


class PropertyExposeDetails(BasePropertyDocumentDetails):
    """Schema for extracting details from a Property Exposé document.

    Inherits common property fields from BasePropertyDocumentDetails and adds
    fields specific to property exposés.
    """

    realty_features: Optional[
        List[
            Literal[
                "lift",
                "basement",
                "garage",
                "loggia",
                "balcony",
                "attic",
                "terrace",
                "pool",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "List of features the property has. Select any that apply, translate to "
            "English if necessary."
        ),
    )
    realty_features_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box for realty features, including page number.",
    )

    surface: Optional[float] = Field(
        default=None, description="Surface of the plot area in square meters."
    )
    surface_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box for plot area surface, including page number.",
    )

    price: Optional[float] = Field(
        default=None, description="Purchase price of the property."
    )
    price_bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for price, including page number."
    )


class EPCCertificateDetails(BasePropertyDocumentDetails):
    """Schema for extracting details from an Energy Performance Certificate (EPC).

    Inherits common property fields from BasePropertyDocumentDetails and adds
    fields specific to EPCs.
    """

    epc_date: Optional[date] = Field(
        default=None,
        description="Date of certificate issue. Also called date of the visit.",
    )
    epc_date_bbox: Optional[BoundingBox] = Field(
        default=None,
        description=(
            "Bounding box for Energy Performance Certificate (EPC) date, including "
            "page number."
        ),
    )

    epc_date_valid_until: Optional[date] = Field(
        default=None,
        description=(
            "Date until which the certificate is valid. Also called Energy "
            "Performance Certificate (EPC) certificate expiration date."
        ),
    )
    epc_date_valid_until_bbox: Optional[BoundingBox] = Field(
        default=None,
        description=(
            "Bounding box for Energy Performance Certificate (EPC) date valid until, "
            "including page number."
        ),
    )
