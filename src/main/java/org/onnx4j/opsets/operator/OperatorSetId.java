/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.operator;

import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.onnx4j.prototypes.OnnxProto3.OperatorSetIdProto;

public class OperatorSetId {

	public static final String ID_SEPARATOR = ":";

	public static final String DEFAULT_DOMAIN = "ai.onnx";

	public static final long DEFAULT_VERSION = 1L;

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
				opsetIds[n] = new OperatorSetId(proto.getDomain(), proto.getVersion());
			}
		} else {
			//
			// Set default opset("ai.onnx:v1") if OperatorSetIdProto's list is
			// empty
			//
			opsetIds = new OperatorSetId[1];
			opsetIds[0] = new OperatorSetId(DEFAULT_DOMAIN, DEFAULT_VERSION);
		}
		return opsetIds;
	}

	public OperatorSetId(String domain, long opsetVersion) {
		super();
		this.domain = StringUtils.isEmpty(domain) ? DEFAULT_DOMAIN : domain;
		this.opsetVersion = opsetVersion;
		this.id = this.domain + ID_SEPARATOR + this.opsetVersion;
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