package org.onnx4j.opsets;

import java.util.List;

import org.onnx4j.onnx.prototypes.OnnxProto3.OperatorSetIdProto;

public class OperatorSetId {
	
	protected String id;

	//
	// The domain of the operator set. Must be unique among all sets.
	//
	protected String domain;

	//
	// The version of the set of operators.
	//
	// The operator set version is a simple integer value that is monotonically
	// increased as new versions of the operator set are published.
	//
	protected long opsetVersion;

	public static OperatorSetId[] from(List<OperatorSetIdProto> protoList) {
		assert protoList != null;
		
		OperatorSetId[] opsetIds;
		
		if (protoList.size() > 0) {
			opsetIds = new OperatorSetId[protoList.size()];
			for (int n = 0; n < opsetIds.length; n++) {
				OperatorSetIdProto proto = protoList.get(n);
				opsetIds[n] = new OperatorSetId(
						proto.getDomain(), proto.getVersion());
			}
		} else {
			//
			// Set default opset("ai.onnx:v1") if OperatorSetIdProto's list is empty
			//
			opsetIds = new OperatorSetId[1];
			opsetIds[0] = new OperatorSetId("", 1L);
		}
		return opsetIds;
	}
	
	public OperatorSetId(String domain, long opsetVersion) {
		super();
		this.domain = domain;
		this.opsetVersion = opsetVersion;
		this.id = domain + ":" + opsetVersion;
	}

	public String getId() {
		return id;
	}

	public String getDomain() {
		return domain;
	}

	public long getOpsetVersion() {
		return opsetVersion;
	}

}
