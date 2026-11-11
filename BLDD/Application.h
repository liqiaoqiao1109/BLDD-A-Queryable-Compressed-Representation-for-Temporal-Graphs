#pragma once

enum class RunMode {
    CompressSave,
    CompressQuery,
    LoadQuery
};

enum class DatasetId : unsigned int {
    Yahoo = 0,
    Flickr,
    EditWiki,
    CbtPow,
    CbtComm,
    WikiLink,
    Sample
};

struct ApplicationOptions {
    RunMode mode;
    DatasetId dataset;
};

// Command-line values override the defaults selected in Main.cpp.
int RunApplication(
    int argc,
    char* argv[],
    const ApplicationOptions& defaults);

